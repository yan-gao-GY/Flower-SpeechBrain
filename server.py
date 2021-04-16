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


from argparse import ArgumentParser
import os
from typing import Callable, Dict, Optional, Tuple, List
import torch
import flwr as fl
import socket
from pathlib import Path
import sys
import numpy as np
from functools import reduce
import json
import pickle

from client import SpeechBrainClient, int_model
from flwr.server.strategy.aggregate import aggregate as aggregate_num
from flwr.common import parameters_to_weights, Weights


pars = ArgumentParser(description="FlowerSpeechBrain")
pars.add_argument(
    "--log_host", type=str, help="HTTP log handler host (no default)",
)
pars.add_argument("--data_path", type=str, help="dataset path")
pars.add_argument("--resume", nargs="+", default=["","-1"], help='initializes global weights from a saved model. [path_to_directory, round_number]')
args = pars.parse_args()

Test = False
#Test = True

Cluster = "canada" # [ox, cam, canada]

DEFAULT_SERVER_ADDRESS = "[::]:9000"
if Cluster == "cam":
    SAVE_PATH_PRE = '/home/yg381/rds/hpc-work/results/fr_train_10clients_wer/'
    # Data_Path = '/home/yg381/rds/hpc-work/datasets/commonvoice/fr/cv-corpus-6.0-2020-12-11/fr'
    Data_Path = '/local/cv-corpus-6.0-2020-12-11/fr'
    Flower_Path = "/home/yg381/flower_speechbrain_updating/wer_based"
    Out_dir = SAVE_PATH_PRE + 'recover'
elif Cluster == "ox":
    SAVE_PATH_PRE = 'results/fr_train_10clients/'
    Data_Path = '/datasets/commonvoice/fr/cv-corpus-6.0-2020-12-11/fr'
    Flower_Path = "/nfs-share/yan/flower_speechbrain_updating/wer_based"
    Out_dir = './recover'
else:
    SAVE_PATH_PRE = 'results/fr_train_10clients/'
    Data_Path = args.data_path
    Flower_Path = '/home/parcollt/scratch/yan/flower_speechbrain_updating/wer_based_valid'
    Out_dir = './recover'


if Test:
    Test_Path = Data_Path + "/train_temp.tsv"
    Tr_add_Path = Data_Path + "/train_temp.tsv"
    Min_fit_clients = 1
    Min_available_clients = 1
else:
    Test_Path = Data_Path + "/test.tsv"
    Tr_add_Path = Data_Path + "/train_add.tsv"
    Min_fit_clients = 10
    Min_available_clients = 10

Tr_Path = Data_Path + "/train_temp.tsv"
Dev_Path = Data_Path + "/train_temp.tsv"
NUM_ROUNDS = 10
Local_epochs = 5
Local_batch_size = 8

Resume_Path = None
Resume_Round = -1
Weighted_Strategy = 'wer'  # [num, loss, wer]

# get server IP
hostname=socket.gethostname()
ip=socket.gethostbyname(hostname)
print('Server IP: ', ip)
with open('server_ip.txt', 'w') as ff:
    ff.write(ip)

# class TrainAfterAggregateStrategy(fl.server.strategy.FedAdam):
class TrainAfterAggregateStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[BaseException],
    ) -> Optional[fl.common.Weights]:

        if not results:
            return None
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None
        # Convert results
        key_name = 'train_loss' if Weighted_Strategy == 'loss' else 'wer'
        weights = None

        if Weighted_Strategy == 'num':
            weights_results = [
                (parameters_to_weights(fit_res.parameters), fit_res.num_examples)
                for client, fit_res in results
            ]
            weights =  aggregate_num(weights_results)
        elif Weighted_Strategy == 'loss' or Weighted_Strategy == 'wer':
            weights_results = [
                (parameters_to_weights(fit_res.parameters), fit_res.metrics[key_name])
                for client, fit_res in results
            ]
            weights = aggregate(weights_results, key_name)

        # Train model after aggregation
        if weights is not None:
            print(f"Train model after aggregation")
            save_path = SAVE_PATH_PRE + "add_train_999"
            asr_brain, dataset = int_model(Flower_Path, Tr_add_Path, Tr_Path, Tr_Path, save_path, Data_Path, add_train=True)
            client = SpeechBrainClient(999, 0.0, asr_brain, dataset)

            weights_after_server_side_training = client.train_speech_recogniser(
                server_params=weights,
                epochs=1,
                batch_size=8,
                timeout=100000,
                delay_factor=0,
                add_train=True
            )
            return weights_after_server_side_training
        # return weights


def aggregate(results: List[Tuple[Weights, int]], key_name) -> Weights:
    """Compute weighted average."""
    measure_list = []
    weights_list = []
    for weights, measure in results:
        if key_name == 'wer':
            if measure > 100:
                measure = 100
            measure = 100 - measure
            print('measure', measure)

        elif key_name == 'loss':
            measure = - measure

        measure_list.append(measure)
        weights_list.append(weights)

    measure_list = np.array(measure_list)
    measure_list = torch.from_numpy(measure_list)
    apply_softmax = torch.nn.Softmax(dim=0)
    measure_list = apply_softmax(measure_list)
    measure_list = measure_list.numpy()
    print('measure_list', measure_list)

    weighted_weights = [[layer * measure for layer in weights] for weights, measure in zip(weights_list, measure_list)]
    weights_prime = [
        reduce(np.add, layer_updates) for layer_updates in zip(*weighted_weights)
    ]
    return weights_prime


def my_init_fn(resume: str, rnd: int) -> Callable[[None], None]:
    """Given the dirctory of the experiment results and a round number,
    this function loads the weights previously saved for that round and
    sets them as the server.weights."""
    def f(self) -> None:
        # Server ended __init__(), DO something?

        # storre a serialised version of the config
        # (this will be sent to the VCM once it connects)
        # serialised_config = json.dumps(config)
        # self.config = serialised_config

        if resume:
            print(f"> Loading weights at round {rnd} from {resume}")
            # load weights from pickle
            weights_file = resume+f"/weights_round{rnd}.pkl"
            with open(weights_file, 'rb') as handle:
                weights = pickle.load(handle)

            self.weights = weights
            self.starting_round = rnd + 1

    return f


def my_end_round_fn(output_dir) -> Callable[[Dict], None]:

    def f(self, args: Dict) -> None:

        rnd = args["current_round"]
        write_mode = 'w' if rnd == 1 else 'a'

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        ## Saving all args to txt and pickles for easy post-training analysis/viz

        # we exclude the weights
        weights = args.pop('weights')  # we don't want to write this to .txt

        # append args to pickle
        output_file = Path(output_dir + "/results.json")
        with open(str(output_file), write_mode) as file:
            # string = [k+"="+str(v) for k, v in args.items()]
            json.dump(args, file)
            file.write("\n")

        output_file = Path(output_dir + "/results.pkl")
        data = []

        # file won't exist in first round
        if output_file.exists():
            # load existing data in pickle
            with open(output_file, 'rb') as handle:
                data = pickle.load(handle)

        # append new result and save
        data.append(args)
        with open(output_file, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        ## Save the weights (one file per round)
        output_file = output_dir + f"/weights_round{rnd}.pkl"
        with open(output_file, 'wb') as handle:
            pickle.dump(weights, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return f

def main() -> None:
    # Create ClientManager & Strategy
    client_manager = fl.server.SimpleClientManager()

    timeout = 200

    # resume_path = args.resume[0]
    # resume_round = int(args.resume[1])

    resume_path = Resume_Path
    resume_round = Resume_Round

    # strategy = fl.server.strategy.FedAvg(
    #     fraction_fit=1,
    #     min_fit_clients=Min_fit_clients,
    #     min_available_clients=Min_available_clients,
    #     eval_fn=evaluate,
    #     on_fit_config_fn=get_on_fit_config_fn(0.01, timeout)
    # )

    # strategy = fl.server.strategy.FedAdam(
    #     fraction_fit=1,
    #     min_fit_clients=Min_fit_clients,
    #     min_available_clients=Min_available_clients,
    #     eval_fn=evaluate,
    #     on_fit_config_fn=get_on_fit_config_fn(0.01, timeout),
    #     current_weights=None
    # )

    # strategy = TrainAfterAggregateStrategy(
    #     fraction_fit=1,
    #     min_fit_clients=Min_fit_clients,
    #     min_available_clients=Min_available_clients,
    #     eval_fn=evaluate,
    #     on_fit_config_fn=get_on_fit_config_fn(0.01, timeout),
    #     current_weights=None
    # )

    strategy = TrainAfterAggregateStrategy(
        fraction_fit=1,
        min_fit_clients=Min_fit_clients,
        min_available_clients=Min_available_clients,
        eval_fn=evaluate,
        on_fit_config_fn=get_on_fit_config_fn(0.01, timeout)
    )


    # Configure logger
    fl.common.logger.configure("server", host=args.log_host)

    server = fl.server.Server(client_manager=client_manager,
                              strategy=strategy,
                              on_init_fn=my_init_fn(resume_path, resume_round),
                              on_round_end_fn=my_end_round_fn(Out_dir))

    # server = fl.server.Server(client_manager=client_manager, strategy=strategy)

    # Run server
    fl.server.start_server(
        DEFAULT_SERVER_ADDRESS, server, config={"num_rounds": NUM_ROUNDS}, grpc_max_message_length=1024*1024*1024
    )

def evaluate(weights: fl.common.Weights) -> Optional[Tuple[float, float]]:
    """Use entire test set for evaluation."""
    data_path = Data_Path
    flower_path = Flower_Path
    tr_path = Tr_Path
    dev_path = Dev_Path
    test_path = Test_Path
    save_path = SAVE_PATH_PRE + "evaluation_199"

    # int model
    asr_brain, dataset = int_model(flower_path, tr_path, dev_path, test_path, save_path, data_path, evaluate=True)

    client = SpeechBrainClient(199, 0.0, asr_brain, dataset)

    # np.save('{}eval_model.npy'.format(SAVE_PATH_PRE), weights, allow_pickle=True)

    nb_ex, lss, acc = client.train_speech_recogniser(
        server_params=weights,
        epochs=1,
        batch_size=8,
        timeout=100000,
        evaluate=True,
        delay_factor=0
    )

    return lss, acc

def get_on_fit_config_fn(
    lr_initial: float, timeout: int
) -> Callable[[int], Dict[str, str]]:
    """Return a function which returns training configurations."""

    def fit_config(rnd: int) -> Dict[str, str]:
        """Return a configuration with static batch size and (local) epochs."""
        config = {
            "epoch_global": str(rnd),
            "epochs": str(Local_epochs),
            "batch_size": str(Local_batch_size),
            "timeout": str(timeout),
        }
        return config

    return fit_config


if __name__ == "__main__":
    main()
