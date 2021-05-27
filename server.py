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
from typing import Callable, Dict, Optional, Tuple, List
import torch
import flwr as fl
import socket
import numpy as np
from functools import reduce

from client import SpeechBrainClient, int_model
from flwr.server.strategy.aggregate import aggregate as aggregate_num
from flwr.common import parameters_to_weights, Weights


parser = ArgumentParser(description="FlowerSpeechBrain")
parser.add_argument(
    "--log_host", type=str, help="HTTP log handler host (no default)",
)
parser.add_argument("--data_path", type=str, help="dataset path")
parser.add_argument("--save_path", type=str, default="./results/", help="path for output files")
parser.add_argument("--config_path", type=str, default="./config/", help="path to config directory")
parser.add_argument("--server_address", type=str, default="[::]:8080", help="server address")
parser.add_argument("--tr_path", type=str, help="train set path")
parser.add_argument("--test_path", type=str, help="test set path")
parser.add_argument("--tr_add_path", type=str, help="additional train set path on the server side")
parser.add_argument("--min_fit_clients", type=int, default=10, help="minimum fit clients")
parser.add_argument("--min_available_clients", type=int, default=10, help="minmum available clients")
parser.add_argument("--rounds", type=int, default=30, help="global training rounds")
parser.add_argument("--local_epochs", type=int, default=5, help="local epochs on each client")
parser.add_argument("--local_batch_size", type=int, default=8, help="local batch size on each client")
parser.add_argument("--weight_strategy", type=str, default="num", help="strategy of weighting clients in [num, loss, wer]")
parser.add_argument('--config_file', type=str, default="CRDNN.yaml", help='config file name')
parser.add_argument('--tokenizer_path', type=str, default=None,
                        help='path for tokenizer (generated from the data for pre-trained model)')
args = parser.parse_args()


# get server IP
hostname=socket.gethostname()
ip=socket.gethostbyname(hostname)
print('Server IP: ', ip)
with open('server_ip.txt', 'w') as ff:
    ff.write(ip)


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
        key_name = 'train_loss' if args.weight_strategy == 'loss' else 'wer'
        weights = None

        if args.weight_strategy == 'num':
            weights_results = [
                (parameters_to_weights(fit_res.parameters), fit_res.num_examples)
                for client, fit_res in results
            ]
            weights =  aggregate_num(weights_results)
        elif args.weight_strategy == 'loss' or args.weight_strategy == 'wer':
            weights_results = [
                (parameters_to_weights(fit_res.parameters), fit_res.metrics[key_name])
                for client, fit_res in results
            ]
            weights = aggregate(weights_results, key_name)

        # Train model after aggregation
        if weights is not None:
            print(f"Train model after aggregation")
            save_path = args.save_path + "add_train_9999"
            asr_brain, dataset = int_model(args.config_path, args.tr_add_path, args.tr_path, args.tr_path, save_path,
                                           args.data_path, args.config_file, args.tokenizer_path, add_train=True)
            client = SpeechBrainClient(9999, 0.0, asr_brain, dataset)

            weights_after_server_side_training = client.train_speech_recogniser(
                server_params=weights,
                epochs=1,
                batch_size=8,
                timeout=100000,
                delay_factor=0,
                add_train=True
            )
            return weights_after_server_side_training


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


def main() -> None:
    # Create ClientManager & Strategy
    client_manager = fl.server.SimpleClientManager()

    timeout = 200

    strategy = TrainAfterAggregateStrategy(
        fraction_fit=1,
        min_fit_clients=args.min_fit_clients,
        min_available_clients=args.min_available_clients,
        eval_fn=evaluate,
        on_fit_config_fn=get_on_fit_config_fn(0.01, timeout)
    )


    # Configure logger
    fl.common.logger.configure("server", host=args.log_host)

    server = fl.server.Server(client_manager=client_manager, strategy=strategy)

    # Run server
    fl.server.start_server(
        args.server_address, server, config={"num_rounds": args.rounds}, grpc_max_message_length=1024*1024*1024
    )

def evaluate(weights: fl.common.Weights):
    """Use entire test set for evaluation."""
    data_path = args.data_path
    flower_path = args.config_path
    tr_path = args.tr_path
    dev_path = tr_path
    test_path = args.test_path
    save_path = args.save_path + "evaluation_19999"

    # int model
    asr_brain, dataset = int_model(flower_path, tr_path, dev_path, test_path, save_path, data_path, args.config_file, args.tokenizer_path, evaluate=True)

    client = SpeechBrainClient(19999, 0.0, asr_brain, dataset)

    nb_ex, lss, acc = client.train_speech_recogniser(
        server_params=weights,
        epochs=1,
        batch_size=8,
        timeout=100000,
        evaluate=True,
        delay_factor=0
    )

    return lss, {"accuracy": acc}

def get_on_fit_config_fn(
    lr_initial: float, timeout: int
) -> Callable[[int], Dict[str, str]]:
    """Return a function which returns training configurations."""

    def fit_config(rnd: int) -> Dict[str, str]:
        """Return a configuration with static batch size and (local) epochs."""
        config = {
            "epoch_global": str(rnd),
            "epochs": str(args.local_epochs),
            "batch_size": str(args.local_batch_size),
            "timeout": str(timeout),
        }
        return config

    return fit_config


if __name__ == "__main__":
    main()
