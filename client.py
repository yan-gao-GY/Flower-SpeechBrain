# Copyright 2020 The Flower Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


import torchaudio
import csv
import logging
import os
import re
import sys
from argparse import ArgumentParser
from typing import Dict
import timeit
import flwr as fl
import numpy as np
from hyperpyyaml import load_hyperpyyaml
import speechbrain as sb
from speechbrain.utils.distributed import run_on_main
from speechbrain.tokenizers.SentencePiece import SentencePiece
import torch
from torch.utils.data import DataLoader
from collections import OrderedDict
torch.set_num_threads(10)

from acoustic_training import (
    ASR,
    set_weights,
    get_weights,
    Stage
)
from flwr.common import EvaluateIns, EvaluateRes, FitIns, FitRes, ParametersRes, Weights


pars = ArgumentParser(description="FlowerSpeechBrain")
pars.add_argument(
    "--log_host", type=str, help="HTTP log handler host (no default)",
)
pars.add_argument("--cid", type=str, help="Client CID (no default)")
pars.add_argument("--delay_factor", type=float, default=0.0, help="Client delay factor (no default)")
pars.add_argument("--data_path", type=str, help="dataset path")
args = pars.parse_args()

Test = False
#Test = True

Cluster = "canada" # [ox, cam, canada]

if Cluster == "cam":
    DEFAULT_SERVER_ADDRESS = "10.43.6.43:9000"
    # Data_Path = '/home/yg381/rds/hpc-work/datasets/commonvoice/fr/cv-corpus-6.0-2020-12-11/fr'
    Data_Path = '/local/cv-corpus-6.0-2020-12-11/fr'
    SAVE_PATH_PRE = '/home/yg381/rds/hpc-work/results/fr_train_10clients_wer/'
    Flower_Path = "/home/yg381/flower_speechbrain_updating/wer_based"
    Pre_train_model_path = "/home/yg381/rds/hpc-work/pre-trained_fl/model.ckpt"
    Tokenizer_Path = "/rds/user/yg381/hpc-work/results/CRDNN_fr_pre_wholeToken/1234/save/train.csv"
elif Cluster == "ox":
    # DEFAULT_SERVER_ADDRESS = "163.1.88.76:9000" # tarawera
    DEFAULT_SERVER_ADDRESS = "163.1.88.101:9000" # ngongotaha
    # DEFAULT_SERVER_ADDRESS = "163.1.88.85:9000"  # mauao
    Data_Path = '/datasets/commonvoice/fr/cv-corpus-6.0-2020-12-11/fr'
    SAVE_PATH_PRE = 'results/fr_train_10clients/'
    Flower_Path = "/nfs-share/yan/flower_speechbrain_updating/wer_based"
    Pre_train_model_path = "/nfs-share/yan/flower_speechbrain_updating/speechbrain/recipes/CommonVoice/ASR/seq2seq/train/results/CRDNN_it/1234/save/CKPT+2021-02-03+10-46-24+00/model.ckpt"
    Tokenizer_Path = "/nfs-share/yan/train_csv/CRDNN_fr_pre/train.csv"
else:
    DEFAULT_SERVER_ADDRESS = "10.70.15.5:9000"
    Data_Path = args.data_path
    SAVE_PATH_PRE = 'results/fr_train_10clients/'
    Flower_Path = '/home/parcollt/scratch/yan/flower_speechbrain_updating/wer_based_valid'
    Pre_train_model_path = '/home/parcollt/scratch/yan/pre-trained_fl/model.ckpt'
    Tokenizer_Path = "/home/parcollt/scratch/yan/train_csv/CRDNN_fr_pre/train.csv"


TR_PATH_PRE = '/split_10_clients/'
Dev_path = Data_Path + "/train_temp.tsv" if Test else Data_Path + "/dev.tsv"
Config_Path = "/configs/CRDNN.yaml"
Num_gpu_evaluate = 4
Eval_device = 'cuda:1'

K1 = "enc.DNN.block_0.norm.norm.num_batches_tracked"
K2 = "enc.DNN.block_1.norm.norm.num_batches_tracked"

class SpeechBrainClient(fl.client.Client):
    def __init__(self,
        cid: int,
        delay_factor: float,
        asr_brain,
        dataset):

        self.cid = cid
        self.params = asr_brain.hparams
        self.modules = asr_brain.modules
        self.asr_brain = asr_brain
        self.dataset = dataset

        # self.new_weights = [ nda for nda in np.load(
        #     self.data_path + "/flower/pretrained_model.npy", allow_pickle=True
        # )]
        self.delay_factor = delay_factor
        fl.common.logger.log(logging.DEBUG, "Starting client %s with delay factor of %s", cid, delay_factor)


    def get_parameters(self) -> ParametersRes:
        print(f"Client {self.cid}: get_parameters")

        weights: Weights = get_weights(self.modules)
        parameters = fl.common.weights_to_parameters(weights)
        return ParametersRes(parameters=parameters)


    def fit(self, ins: FitIns) -> FitRes:
        print(f"Client {self.cid}: fit")

        weights: Weights = fl.common.parameters_to_weights(ins.parameters)
        config = ins.config

        # Read training configuration
        global_rounds = int(config["epoch_global"])
        print("Current global round: ", global_rounds)
        epochs = int(config["epochs"])
        batch_size = int(config["batch_size"])
        timeout = int(config["timeout"])

        (
            new_weights,
            num_examples,
            num_examples_ceil,
            fit_duration,
            avg_loss,
            avg_wer
        ) = self.train_speech_recogniser(
            weights,
            epochs,
            batch_size,
            timeout,
            delay_factor = self.delay_factor,
            global_rounds=global_rounds
        )

        # np.save('pretrained_model.npy', self.new_weights, allow_pickle=True)

        fl.common.logger.log(logging.DEBUG, "client %s had fit_duration %s with %s of %s", self.cid, fit_duration, num_examples, num_examples_ceil)

        metrics = {"train_loss": avg_loss, "wer": avg_wer}

        return FitRes(
            parameters=self.get_parameters().parameters,
            num_examples=num_examples,
            num_examples_ceil=num_examples_ceil,
            fit_duration=fit_duration,
            metrics=metrics
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        print(f"Client {self.cid}: evaluate")

        weights = fl.common.parameters_to_weights(ins.parameters)
        config = ins.config

        epochs = int(config["epochs"])
        batch_size = int(config["batch_size"])

        num_examples, loss, cer = self.train_speech_recogniser(
            server_params=weights,
            epochs=epochs,
            batch_size=batch_size,
            timeout=100000,
            evaluate=True,
        )

        # Return the number of evaluation examples and the evaluation result (loss)
        return EvaluateRes(
            num_examples=num_examples, loss=float(loss), accuracy=float(cer)
        )

    def train_speech_recogniser(
        self,
        server_params,
        epochs,
        batch_size,
        timeout,
        evaluate=False,
        delay_factor=0.0,
        add_train=False,
        global_rounds=None
    ):

        # print('Global Rounds:', global_rounds)
        # self.params.batch_size = batch_size
        self.params.epoch_counter.limit = epochs
        self.params.epoch_counter.current = 0

        # if self.cid == 0:
        #     self.params.device = "cuda:0"

        train_data, valid_data, test_data = self.dataset

        # Set the parameters to the ones given by the server
        if server_params is not None:
            set_weights(server_params, self.modules, evaluate, add_train, self.params.device)

        if global_rounds == 1 and not add_train and not evaluate:
            print("loading pre-trained model...")
            state_dict = torch.load(Pre_train_model_path)
            self.params.model.load_state_dict(state_dict)

            # if self.asr_brain.checkpointer is not None:
            #     print("loading pre-trained model from checkpoint...")
            #     self.asr_brain.checkpointer.recover_if_possible(
            #         device=torch.device(self.asr_brain.device)
            #     )

        if global_rounds != 1:
            state_dict_norm = OrderedDict()
            state_dict_norm[K1] = torch.tensor(1, device=self.params.device)
            state_dict_norm[K2] = torch.tensor(0, device=self.params.device)
            self.modules.load_state_dict(state_dict_norm, strict=False)

        # Load best checkpoint for evaluation
        if evaluate:
            self.params.wer_file = self.params.output_folder + "/wer_test.txt"
            batch_count, loss, wer = self.asr_brain.evaluate(
                test_data,
                min_key="WER",
                test_loader_kwargs=self.params.test_dataloader_options,
            )

            # if not isinstance(test_data, DataLoader):
            #     self.params.test_dataloader_options["ckpt_prefix"] = None
            #     test_set = self.asr_brain.make_dataloader(
            #         test_data, Stage.TEST, **self.params.test_dataloader_options
            #     )

            return batch_count, loss, wer

        # Training
        fit_begin = timeit.default_timer()
        # if add_train:
        #     count_sample = self.asr_brain.fit(
        #         self.params.epoch_counter,
        #         train_data,
        #         valid_data,
        #         train_loader_kwargs=self.params.dataloader_options,
        #         valid_loader_kwargs=self.params.test_dataloader_options,
        #     )
        # else:
        #     count_sample = 95

        count_sample, avg_loss, avg_wer = self.asr_brain.fit(
            self.params.epoch_counter,
            train_data,
            valid_data,
            train_loader_kwargs=self.params.dataloader_options,
            valid_loader_kwargs=self.params.test_dataloader_options,
        )
        # count_sample = 95


        # retrieve the parameters to return
        params_list = get_weights(self.modules)

        if add_train:
            return params_list

        fit_duration = timeit.default_timer() - fit_begin

        # Manage when last batch isn't full w.r.t batch size
        train_set = sb.dataio.dataloader.make_dataloader(train_data, **self.params.dataloader_options)
        if count_sample > len(train_set) * self.params.batch_size * epochs:
            count_sample = len(train_set) * self.params.batch_size * epochs

        return (
            params_list,
            count_sample,
            # len(train_set) * self.params.batch_size * epochs,
            len(train_set) * self.params.batch_size * epochs,
            fit_duration,
            avg_loss,
            avg_wer
        )

# Define custom data procedure
def data_io_prepare(hparams):

    # 1. Define datasets
    data_folder = hparams["data_folder"]

    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_csv"], replacements={"data_root": data_folder},
    )

    if hparams["sorting"] == "ascending":
        # we sort training data to speed up training and get better results.
        train_data = train_data.filtered_sorted(
            sort_key="duration",
            key_max_value={"duration": hparams["avoid_if_longer_than"]},
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["dataloader_options"]["shuffle"] = False

    elif hparams["sorting"] == "descending":
        train_data = train_data.filtered_sorted(
            sort_key="duration",
            reverse=True,
            key_max_value={"duration": hparams["avoid_if_longer_than"]},
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["dataloader_options"]["shuffle"] = False

    elif hparams["sorting"] == "random":
        pass

    else:
        raise NotImplementedError(
            "sorting must be random, ascending or descending"
        )

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["valid_csv"], replacements={"data_root": data_folder},
    )
    # We also sort the validation data so it is faster to validate
    valid_data = valid_data.filtered_sorted(sort_key="duration")

    test_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["test_csv"], replacements={"data_root": data_folder},
    )
    # We also sort the test data so it is faster to validate
    test_data = test_data.filtered_sorted(sort_key="duration")

    datasets = [train_data, valid_data, test_data]

    # defining tokenizer and loading it
    tokenizer = SentencePiece(
        model_dir=hparams["save_folder"],
        vocab_size=hparams["output_neurons"],
        csv_train=hparams["tokenizer_csv"],
        csv_read="wrd",
        model_type=hparams["token_type"],
        character_coverage=hparams["character_coverage"],
    )

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        info = torchaudio.info(wav)
        sig = sb.dataio.dataio.read_audio(wav)
        resampled = torchaudio.transforms.Resample(
            info.sample_rate, hparams["sample_rate"],
        )(sig)
        return resampled

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("wrd")
    @sb.utils.data_pipeline.provides(
        "tokens_list", "tokens_bos", "tokens_eos", "tokens"
    )
    def text_pipeline(wrd):
        tokens_list = tokenizer.sp.encode_as_ids(wrd)
        yield tokens_list
        tokens_bos = torch.LongTensor([hparams["bos_index"]] + (tokens_list))
        yield tokens_bos
        tokens_eos = torch.LongTensor(tokens_list + [hparams["eos_index"]])
        yield tokens_eos
        tokens = torch.LongTensor(tokens_list)
        yield tokens

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets, ["id", "sig", "tokens_bos", "tokens_eos", "tokens"],
    )
    return train_data, valid_data, test_data, tokenizer

def int_model(
    flower_path,
    tr_path,
    dev_path,
    test_path,
    save_path,
    data_path,
    evaluate=False,
    add_train=False):

    # This hack needed to import data preparation script from ..
    # current_dir = os.path.dirname(os.path.abspath(__file__))
    # sys.path.append(os.path.dirname(current_dir))

    # Load hyperparameters file with command-line overrides
    params_file = flower_path + Config_Path

    # Override with FLOWER PARAMS
    if evaluate:
        overrides = {
            "output_folder": save_path,
            "number_of_epochs": 1,
            "test_batch_size": 16,
            "device": Eval_device,
            # "device": 'cpu'
        }
    elif add_train:
        overrides = {
            "output_folder": save_path,
            "lr": 0.01
        }

    else:
        overrides = {
            "output_folder": save_path
        }

    # if evaluate or add_train:
    #     run_opts = {'debug': False,
    #                 'debug_batches': 2,
    #                 'debug_epochs': 2,
    #                 'device': 'cuda:0',
    #                 'data_parallel_count': Num_gpu_evaluate,
    #                 'data_parallel_backend': True,
    #                 'distributed_launch': False,
    #                 'distributed_backend': 'nccl'}
    # else:
    #     run_opts = None
    run_opts = None

    with open(params_file) as fin:
        params = load_hyperpyyaml(fin, overrides)

    params["data_folder"] = data_path
    params["train_tsv_file"] = tr_path
    params["dev_tsv_file"] = dev_path
    params["test_tsv_file"] = test_path
    params["save_folder"] = params["output_folder"] + "/save"
    params["train_csv"] = params["save_folder"] + "/train.csv"
    params["valid_csv"] = params["save_folder"] + "/dev.csv"
    params["test_csv"] = params["save_folder"] + "/test.csv"
    params["tokenizer_csv"] = Tokenizer_Path

    # Dataset preparation (parsing CommonVoice)
    from common_voice_prepare import prepare_common_voice  # noqa

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=params["output_folder"],
        hyperparams_to_save=params_file,
        overrides=overrides,
    )

    # Due to DDP, we do the preparation ONLY on the main python process
    run_on_main(
        prepare_common_voice,
        kwargs={
            "data_folder": params["data_folder"],
            "save_folder": params["save_folder"],
            "train_tsv_file": params["train_tsv_file"],
            "dev_tsv_file": params["dev_tsv_file"],
            "test_tsv_file": params["test_tsv_file"],
            "accented_letters": params["accented_letters"],
            "language": params["language"],
        },
    )

    # Create the datasets objects as well as tokenization and encoding :-D
    train_data, valid_data, test_data, tokenizer = data_io_prepare(params)

    # Trainer initialization
    asr_brain = ASR(
        modules=params["modules"],
        hparams=params,
        run_opts=run_opts,
        opt_class=params["opt_class"],
        checkpointer=params["checkpointer"],
    )

    # Adding objects to trainer.
    asr_brain.tokenizer = tokenizer

    return asr_brain, [train_data, valid_data, test_data]


def main() -> None:
    # Configure logger
    fl.common.logger.configure(f"client_{args.cid}", host=args.log_host)

    # initialise path
    tr_path = Data_Path + "/train_temp.tsv" if Test else Data_Path + TR_PATH_PRE + "train_" + str(args.cid) + ".tsv"
    #tr_path = Data_Path + "/train.tsv"
    test_path = Dev_path  # We only evaluate on small dataset on clients
    dev_path = Dev_path  # We only evaluate on small dataset on clients
    save_path = SAVE_PATH_PRE + "client_" + str(args.cid)

    # int model
    asr_brain, dataset = int_model(Flower_Path, tr_path, dev_path, test_path, save_path, Data_Path)

    # Start client
    client = SpeechBrainClient(args.cid, args.delay_factor, asr_brain, dataset)

    #client.fit( (client.get_parameters().parameters, config))
    fl.client.start_client(DEFAULT_SERVER_ADDRESS, client, grpc_max_message_length=1024*1024*1024)


if __name__ == "__main__":
    main()
