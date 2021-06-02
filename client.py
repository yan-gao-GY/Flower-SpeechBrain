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
import logging
from argparse import ArgumentParser
import timeit
import flwr as fl
from hyperpyyaml import load_hyperpyyaml
import speechbrain as sb
from speechbrain.utils.distributed import run_on_main
from speechbrain.tokenizers.SentencePiece import SentencePiece
import torch
from collections import OrderedDict
from math import exp
torch.set_num_threads(8)

from acoustic_training import (
    ASR,
    set_weights,
    get_weights
)
from flwr.common import EvaluateIns, EvaluateRes, FitIns, FitRes, ParametersRes, Weights


class SpeechBrainClient(fl.client.Client):
    def __init__(self,
        cid: int,
        asr_brain,
        dataset,
        pre_train_model_path=None):

        self.cid = cid
        self.params = asr_brain.hparams
        self.modules = asr_brain.modules
        self.asr_brain = asr_brain
        self.dataset = dataset
        self.pre_train_model_path = pre_train_model_path

        # self.new_weights = [ nda for nda in np.load(
        #     self.data_path + "/flower/pretrained_model.npy", allow_pickle=True
        # )]
        fl.common.logger.log(logging.DEBUG, "Starting client %s", cid)


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
            global_rounds=global_rounds
        )

        # np.save('pretrained_model.npy', self.new_weights, allow_pickle=True)

        fl.common.logger.log(logging.DEBUG, "client %s had fit_duration %s with %s of %s", self.cid, fit_duration, num_examples, num_examples_ceil)

        metrics = {"train_loss": avg_loss, "wer": avg_wer}

        torch.cuda.empty_cache()

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

        # config = ins.config
        # epochs = int(config["epochs"])
        # batch_size = int(config["batch_size"])

        num_examples, loss, wer = self.train_speech_recogniser(
            server_params=weights,
            epochs=1,
            evaluate=True
        )
        torch.cuda.empty_cache()

        # Return the number of evaluation examples and the evaluation result (loss)
        return EvaluateRes(
            num_examples=num_examples, loss=float(loss), accuracy=float(wer)
        )

    def train_speech_recogniser(
        self,
        server_params,
        epochs,
        evaluate=False,
        add_train=False,
        global_rounds=None
    ):

        # print('Global Rounds:', global_rounds)
        self.params.epoch_counter.limit = epochs
        self.params.epoch_counter.current = 0

        train_data, valid_data, test_data = self.dataset

        # Set the parameters to the ones given by the server
        if server_params is not None:
            set_weights(server_params, self.modules, evaluate, add_train, self.params.device)

        if global_rounds == 1 and not add_train and not evaluate:
            if self.pre_train_model_path is not None:
                print("loading pre-trained model...")
                state_dict = torch.load(self.pre_train_model_path)
                self.params.model.load_state_dict(state_dict)

            # if self.asr_brain.checkpointer is not None:
            #     print("loading pre-trained model from checkpoint...")
            #     self.asr_brain.checkpointer.recover_if_possible(
            #         device=torch.device(self.asr_brain.device)
            #     )

        if global_rounds != 1:
            # two layer names that do not join aggregation
            k1 = "enc.DNN.block_0.norm.norm.num_batches_tracked"
            k2 = "enc.DNN.block_1.norm.norm.num_batches_tracked"

            state_dict_norm = OrderedDict()
            state_dict_norm[k1] = torch.tensor(1, device=self.params.device)
            state_dict_norm[k2] = torch.tensor(0, device=self.params.device)
            self.modules.load_state_dict(state_dict_norm, strict=False)

        # Load best checkpoint for evaluation
        if evaluate:
            self.params.wer_file = self.params.output_folder + "/wer_test.txt"
            batch_count, loss, wer = self.asr_brain.evaluate(
                test_data,
                # min_key="WER",
                test_loader_kwargs=self.params.test_dataloader_options,
            )
            return batch_count, loss, wer

        # Training
        fit_begin = timeit.default_timer()

        count_sample, avg_loss, avg_wer = self.asr_brain.fit(
            self.params.epoch_counter,
            train_data,
            valid_data,
            train_loader_kwargs=self.params.dataloader_options,
            valid_loader_kwargs=self.params.test_dataloader_options,
        )

        # exp operation to avg_loss and avg_wer
        avg_wer = 100 if avg_wer > 100 else avg_wer
        avg_loss = exp(- avg_loss)
        avg_wer = exp(100 - avg_wer)

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
            len(train_set) * self.params.batch_size * epochs,
            fit_duration,
            avg_loss,
            avg_wer
        )

# Define custom data procedure
def dataio_prepare(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions."""

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

    # We also sort the validation data so it is faster to validate
    test_data = test_data.filtered_sorted(sort_key="duration")

    datasets = [train_data, valid_data, test_data]

    # defining tokenizer and loading it
    tokenizer = SentencePiece(
        model_dir=hparams["save_folder"],
        vocab_size=hparams["output_neurons"],
        annotation_train=hparams["tokenizer_csv"],
        annotation_read="wrd",
        model_type=hparams["token_type"],
        character_coverage=hparams["character_coverage"],
    )

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        info = torchaudio.info(wav)
        sig = sb.dataio.dataio.read_audio(wav)
        if info.num_channels > 1:
            sig = torch.mean(sig, dim=1)
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
    config_file="CRDNN.yaml",
    tokenizer_path=None,
    eval_device="cuda:0",
    evaluate=False,
    add_train=False):

    # Load hyperparameters file with command-line overrides
    params_file = flower_path + config_file

    # Override with FLOWER PARAMS
    if evaluate:
        overrides = {
            "output_folder": save_path,
            "number_of_epochs": 1,
            "test_batch_size": 4,
            "device": eval_device,
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
    params["tokenizer_csv"] = tokenizer_path if tokenizer_path is not None else params["train_csv"]

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
    train_data, valid_data, test_data, tokenizer = dataio_prepare(params)

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
    parser = ArgumentParser(description="FlowerSpeechBrain")
    parser.add_argument(
        "--log_host", type=str, help="HTTP log handler host (no default)",
    )
    parser.add_argument("--cid", type=str, help="Client CID (no default)")
    parser.add_argument("--data_path", type=str, help="dataset path")
    parser.add_argument('--server_address', type=str, default="[::]:8080", help='server IP:PORT')
    parser.add_argument("--tr_path", type=str, help="train set path")
    parser.add_argument('--dev_path', type=str, help='dev set path')
    parser.add_argument('--save_path_pre', type=str, default="./results/", help='path for output files')
    parser.add_argument('--pre_train_model_path', type=str, default=None, help='path for pre-trained model')
    parser.add_argument('--tokenizer_path', type=str, default=None,
                        help='path for tokenizer (generated from the data for pre-trained model)')
    parser.add_argument('--config_path', type=str, default="./configs/", help='path to config directory')
    parser.add_argument('--config_file', type=str, default="CRDNN.yaml", help='config file name')
    parser.add_argument('--eval_device', type=str, default="cuda:0", help='device for evaluation')
    args = parser.parse_args()

    # Configure logger
    fl.common.logger.configure(f"client_{args.cid}", host=args.log_host)

    # initialise path
    tr_path = args.tr_path + "train_" + str(args.cid) + ".tsv"
    test_path = args.dev_path
    dev_path = args.dev_path
    save_path = args.save_path_pre + "client_" + str(args.cid)

    # int model
    asr_brain, dataset = int_model(args.config_path, tr_path, dev_path, test_path, save_path, args.data_path,
                                   args.config_file, args.tokenizer_path, args.eval_device)

    # Start client
    client = SpeechBrainClient(args.cid, asr_brain, dataset, args.pre_train_model_path)

    #client.fit( (client.get_parameters().parameters, config))
    fl.client.start_client(args.server_address, client, grpc_max_message_length=1024*1024*1024)


if __name__ == "__main__":
    main()
