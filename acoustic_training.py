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


import torch
from tqdm.contrib import tqdm
import speechbrain as sb
from speechbrain.utils.data_utils import undo_padding

import flwr as fl
import time
from collections import OrderedDict
import numpy as np
from torch.utils.data import DataLoader
from speechbrain.utils.distributed import run_on_main
from enum import Enum, auto

# two layer names that do not join aggregation
K1 = "enc.DNN.block_0.norm.norm.num_batches_tracked"
K2 = "enc.DNN.block_1.norm.norm.num_batches_tracked"

class Stage(Enum):
    """Simple enum to track stage of experiments."""

    TRAIN = auto()
    VALID = auto()
    TEST = auto()

def set_weights(weights: fl.common.Weights, modules, evaluate, add_train, device) -> None:
    """Set model weights from a list of NumPy ndarrays."""
    state_dict = OrderedDict()
    valid_keys = [k for k in modules.state_dict().keys() if k != K1 and k != K2]
    for k, v in zip(valid_keys, weights):
        v_ = torch.Tensor(np.array(v))
        v_ = v_.to(device)
        state_dict[k] = v_

    modules.load_state_dict(state_dict, strict=False)

def get_weights(modules) -> fl.common.Weights:
    """Get model weights as a list of NumPy ndarrays."""
    w = []
    for k, v in modules.state_dict().items():
        if k != K1 and k != K2:
            w.append(v.cpu().numpy())
    return w

# Define training procedure
class ASR(sb.core.Brain):
    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the output probabilities."""

        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        tokens_bos, _ = batch.tokens_bos
        wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)

        # Forward pass
        feats = self.hparams.compute_features(wavs)
        feats = self.modules.normalize(feats, wav_lens)

        ## Add augmentation if specified
        if stage == sb.Stage.TRAIN:
            if hasattr(self.hparams, "augmentation"):
                feats = self.hparams.augmentation(feats)

        x = self.modules.enc(feats.detach())
        e_in = self.modules.emb(tokens_bos)  # y_in bos + tokens
        h, _ = self.modules.dec(e_in, x, wav_lens)
        # Output layer for seq2seq log-probabilities
        logits = self.modules.seq_lin(h)
        p_seq = self.hparams.log_softmax(logits)

        # Compute outputs
        if stage == sb.Stage.TRAIN:
            p_tokens_gr, scores = self.hparams.greedy_searcher(x, wav_lens)
            current_epoch = self.hparams.epoch_counter.current
            if current_epoch <= self.hparams.number_of_ctc_epochs:
                # Output layer for ctc log-probabilities
                logits = self.modules.ctc_lin(x)
                p_ctc = self.hparams.log_softmax(logits)
                return p_ctc, p_seq, wav_lens, p_tokens_gr
            else:
                return p_seq, wav_lens, p_tokens_gr
        else:
            p_tokens, scores = self.hparams.beam_searcher(x, wav_lens)
            return p_seq, wav_lens, p_tokens

    def fit(
            self,
            epoch_counter,
            train_set,
            valid_set=None,
            progressbar=None,
            train_loader_kwargs={},
            valid_loader_kwargs={},
    ):

        if not isinstance(train_set, DataLoader):
            train_set = self.make_dataloader(
                train_set, stage=sb.Stage.TRAIN, **train_loader_kwargs
            )
        if valid_set is not None and not isinstance(valid_set, DataLoader):
            valid_set = self.make_dataloader(
                valid_set,
                stage=sb.Stage.VALID,
                ckpt_prefix=None,
                **valid_loader_kwargs,
            )

        self.on_fit_start()

        if progressbar is None:
            progressbar = self.progressbar

        batch_count = 0
        # Iterate epochs
        for epoch in epoch_counter:

            # Training stage
            self.on_stage_start(sb.Stage.TRAIN, epoch)
            self.modules.train()
            avg_wer = 0.0

            # Reset nonfinite count to 0 each epoch
            self.nonfinite_count = 0

            if self.train_sampler is not None and hasattr(
                    self.train_sampler, "set_epoch"
            ):
                self.train_sampler.set_epoch(epoch)

            # Time since last intra-epoch checkpoint
            last_ckpt_time = time.time()

            # Only show progressbar if requested and main_process
            enable = progressbar and sb.utils.distributed.if_main_process()
            with tqdm(
                    train_set,
                    initial=self.step,
                    dynamic_ncols=True,
                    disable=not enable,
            ) as t:
                for batch in t:
                    self.step += 1
                    loss, wer = self.fit_batch(batch)
                    _, wav_lens = batch.sig
                    batch_count += wav_lens.shape[0]

                    self.avg_train_loss = self.update_average(
                        loss, self.avg_train_loss
                    )
                    avg_wer = self.update_average_wer(
                        wer, avg_wer
                    )
                    t.set_postfix(train_loss=self.avg_train_loss)

                    # Debug mode only runs a few batches
                    if self.debug and self.step == self.debug_batches:
                        break

                    if (
                            self.checkpointer is not None
                            and self.ckpt_interval_minutes > 0
                            and time.time() - last_ckpt_time
                            >= self.ckpt_interval_minutes * 60.0
                    ):
                        run_on_main(self._save_intra_epoch_ckpt)
                        last_ckpt_time = time.time()

            # Run train "on_stage_end" on all processes
            if epoch == epoch_counter.limit:
                avg_loss = self.avg_train_loss

            self.on_stage_end(sb.Stage.TRAIN, self.avg_train_loss, epoch)
            self.avg_train_loss = 0.0
            self.step = 0

            # Validation stage
            if valid_set is not None:
                self.on_stage_start(sb.Stage.VALID, epoch)
                self.modules.eval()
                avg_valid_loss = 0.0
                with torch.no_grad():
                    for batch in tqdm(
                            valid_set, dynamic_ncols=True, disable=not enable
                    ):
                        self.step += 1
                        loss = self.evaluate_batch(batch, stage=sb.Stage.VALID)
                        avg_valid_loss = self.update_average(
                            loss, avg_valid_loss
                        )

                        # Debug mode only runs a few batches
                        if self.debug and self.step == self.debug_batches:
                            break

                    # Only run validation "on_stage_end" on main process
                    self.step = 0
                    # run_on_main(
                    #     self.on_stage_end,
                    #     args=[sb.Stage.VALID, avg_valid_loss, epoch],
                    # )
                    valid_wer = self.on_stage_end(sb.Stage.VALID, avg_valid_loss, epoch)
                    if epoch == epoch_counter.limit:
                        valid_wer_last = valid_wer

            # Debug mode only runs a few epochs
            if self.debug and epoch == self.debug_epochs:
                break

        return batch_count, avg_loss, valid_wer_last

    def update_average_wer(self, wer, avg_wer):
        avg_wer -= avg_wer / (self.step + 1)
        avg_wer += float(wer) / (self.step + 1)
        return avg_wer

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss (CTC+NLL) given predictions and targets."""

        current_epoch = self.hparams.epoch_counter.current
        if stage == sb.Stage.TRAIN:
            if current_epoch <= self.hparams.number_of_ctc_epochs:
                p_ctc, p_seq, wav_lens, p_tokens_gr = predictions
            else:
                p_seq, wav_lens, p_tokens_gr = predictions
        else:
            p_seq, wav_lens, predicted_tokens = predictions

        ids = batch.id
        tokens_eos, tokens_eos_lens = batch.tokens_eos
        tokens, tokens_lens = batch.tokens

        loss_seq = self.hparams.seq_cost(
            p_seq, tokens_eos, length=tokens_eos_lens
        )

        # Add ctc loss if necessary
        if (
                stage == sb.Stage.TRAIN
                and current_epoch <= self.hparams.number_of_ctc_epochs
        ):
            loss_ctc = self.hparams.ctc_cost(
                p_ctc, tokens, wav_lens, tokens_lens
            )
            loss = self.hparams.ctc_weight * loss_ctc
            loss += (1 - self.hparams.ctc_weight) * loss_seq
        else:
            loss = loss_seq

        if stage == sb.Stage.TRAIN:
            predicted_words = self.tokenizer(
                p_tokens_gr, task="decode_from_list"
            )

            # Convert indices to words
            target_words = undo_padding(tokens, tokens_lens)
            target_words = self.tokenizer(target_words, task="decode_from_list")
            wer_stats = sb.utils.metric_stats.ErrorRateStats()
            wer_stats.append(ids=ids, predict=predicted_words, target=target_words)
            stats = wer_stats.summarize()
            wer = stats['WER']
            return loss, wer

        else:
            # Decode token terms to words
            predicted_words = self.tokenizer(
                predicted_tokens, task="decode_from_list"
            )

            # Convert indices to words
            target_words = undo_padding(tokens, tokens_lens)
            target_words = self.tokenizer(target_words, task="decode_from_list")

            self.wer_metric.append(ids, predicted_words, target_words)
            self.cer_metric.append(ids, predicted_words, target_words)

            return loss

    def fit_batch(self, batch):
        """Train the parameters given a single batch in input"""
        predictions = self.compute_forward(batch, sb.Stage.TRAIN)
        loss, wer= self.compute_objectives(predictions, batch, sb.Stage.TRAIN)
        loss.backward()
        if self.check_gradients(loss):
            self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.detach(), wer

    def evaluate_batch(self, batch, stage):
        """Computations needed for validation/test batches"""
        predictions = self.compute_forward(batch, stage=stage)
        with torch.no_grad():
            loss = self.compute_objectives(predictions, batch, stage=stage)
        return loss.detach()

    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch"""
        if stage != sb.Stage.TRAIN:
            self.cer_metric = self.hparams.cer_computer()
            self.wer_metric = self.hparams.error_rate_computer()

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of an epoch."""
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            stage_stats["CER"] = self.cer_metric.summarize("error_rate")
            stage_stats["WER"] = self.wer_metric.summarize("error_rate")

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:
            old_lr, new_lr = self.hparams.lr_annealing(stage_stats["loss"])
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)
            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": old_lr},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            return stage_stats["WER"]
            # self.checkpointer.save_and_keep_only(
            #     meta={"WER": stage_stats["WER"]}, min_keys=["WER"],
            # )
        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            with open(self.hparams.wer_file, "w") as w:
                self.wer_metric.write_stats(w)

            return stage_stats["WER"]

    def on_fit_start(self):
        # Run this *after* starting all processes since jit modules cannot be
        # pickled.
        self._compile_jit()

        # Wrap modules with parallel backend after jit
        self._wrap_distributed()

        # Initialize optimizers after parameters are configured
        self.init_optimizers()

        # Load latest checkpoint to resume training if interrupted
        if self.checkpointer is not None:
            self.checkpointer.recover_if_possible(
                device=torch.device(self.device)
            )

    def evaluate(
        self,
        test_set,
        progressbar=None,
        test_loader_kwargs={},
    ):
        """Iterate test_set and evaluate brain performance. By default, loads
        the best-performing checkpoint (as recorded using the checkpointer).

        Arguments
        ---------
        test_set : Dataset, DataLoader
            If a DataLoader is given, it is iterated directly. Otherwise passed
            to self.make_dataloader()
        max_key : str
            Key to use for finding best checkpoint, passed to on_evaluate_start
        min_key : str
            Key to use for finding best checkpoint, passed to on_evaluate_start
        progressbar : bool
            Whether to display the progress in a progressbar.
        test_loader_kwargs : dict
            Kwargs passed to `make_dataloader()` if test_set is a Dataset, not
            DataLoader. NOTE: loader_kwargs["ckpt_prefix"] gets automatically
            overwritten to None (so that the test DataLoader is not added to
            the checkpointer).

        Returns
        -------
        average test loss
        """
        if progressbar is None:
            progressbar = self.progressbar

        if not isinstance(test_set, DataLoader):
            test_loader_kwargs["ckpt_prefix"] = None
            test_set = self.make_dataloader(
                test_set, sb.Stage.TEST, **test_loader_kwargs
            )

        self.on_stage_start(sb.Stage.TEST, epoch=None)
        self.modules.eval()
        avg_test_loss = 0.0
        batch_count = 0
        with torch.no_grad():
            for batch in tqdm(
                test_set, dynamic_ncols=True, disable=not progressbar
            ):
                self.step += 1
                _, wav_lens = batch.sig
                batch_count += wav_lens.shape[0]
                loss = self.evaluate_batch(batch, stage=sb.Stage.TEST)
                avg_test_loss = self.update_average(loss, avg_test_loss)

                # Debug mode only runs a few batches
                if self.debug and self.step == self.debug_batches:
                    break

            # Only run evaluation "on_stage_end" on main process
            wer = self.on_stage_end(sb.Stage.TEST, avg_test_loss, None)
        self.step = 0

        return batch_count, avg_test_loss, wer




