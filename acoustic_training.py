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
from torch.utils.data import DataLoader
import speechbrain as sb
from speechbrain.utils.data_utils import undo_padding
from speechbrain.core import AMPConfig
from speechbrain.dataio.dataloader import LoopedLoader
from speechbrain.utils.distributed import run_on_main, if_main_process

import time
from collections import OrderedDict
import numpy as np
from enum import Enum, auto

# two layer names that do not join aggregation
K1 = "enc.DNN.block_0.norm.norm.num_batches_tracked"
K2 = "enc.DNN.block_1.norm.norm.num_batches_tracked"

class Stage(Enum):
    """Simple enum to track stage of experiments."""

    TRAIN = auto()
    VALID = auto()
    TEST = auto()

def set_weights(weights, modules, evaluate, add_train, device) -> None:
    """Set model weights from a list of NumPy ndarrays."""
    state_dict = OrderedDict()
    valid_keys = [k for k in modules.state_dict().keys() if k != K1 and k != K2]
    for k, v in zip(valid_keys, weights):
        v_ = torch.Tensor(np.array(v))
        v_ = v_.to(device)
        state_dict[k] = v_

    modules.load_state_dict(state_dict, strict=False)

def get_weights(modules):
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

        # Add waveform augmentation if specified.
        if stage == sb.Stage.TRAIN and hasattr(self.hparams, "wav_augment"):
            wavs, wav_lens = self.hparams.wav_augment(wavs, wav_lens)
            tokens_bos = self.hparams.wav_augment.replicate_labels(tokens_bos)

        # Forward pass
        feats = self.hparams.compute_features(wavs)
        feats = self.modules.normalize(feats, wav_lens)

        # Add feature augmentation if specified.
        if stage == sb.Stage.TRAIN and hasattr(self.hparams, "fea_augment"):
            feats, fea_lens = self.hparams.fea_augment(feats, wav_lens)
            tokens_bos = self.hparams.fea_augment.replicate_labels(tokens_bos)

        x = self.modules.enc(feats.detach())
        e_in = self.modules.emb(tokens_bos)  # y_in bos + tokens
        h, _ = self.modules.dec(e_in, x, wav_lens)
        # Output layer for seq2seq log-probabilities
        logits = self.modules.seq_lin(h)
        p_seq = self.hparams.log_softmax(logits)

        # Compute outputs
        p_ctc, p_tokens = None, None
        if stage == sb.Stage.TRAIN:
            current_epoch = self.hparams.epoch_counter.current
            if current_epoch <= self.hparams.number_of_ctc_epochs:
                # Output layer for ctc log-probabilities
                logits = self.modules.ctc_lin(x)
                p_ctc = self.hparams.log_softmax(logits)

        p_tokens, _, _, _ = self.hparams.beam_searcher(x, wav_lens)

        return p_ctc, p_seq, wav_lens, p_tokens

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss (CTC+NLL) given predictions and targets."""

        p_ctc, p_seq, wav_lens, predicted_tokens = predictions
        ids = batch.id
        tokens_eos, tokens_eos_lens = batch.tokens_eos
        tokens, tokens_lens = batch.tokens

        if stage == sb.Stage.TRAIN:
            if hasattr(self.hparams, "wav_augment"):
                tokens = self.hparams.wav_augment.replicate_labels(tokens)
                tokens_lens = self.hparams.wav_augment.replicate_labels(
                    tokens_lens
                )
                tokens_eos = self.hparams.wav_augment.replicate_labels(
                    tokens_eos
                )
                tokens_eos_lens = self.hparams.wav_augment.replicate_labels(
                    tokens_eos_lens
                )
            if hasattr(self.hparams, "fea_augment"):
                tokens = self.hparams.fea_augment.replicate_labels(tokens)
                tokens_lens = self.hparams.fea_augment.replicate_labels(
                    tokens_lens
                )
                tokens_eos = self.hparams.fea_augment.replicate_labels(
                    tokens_eos
                )
                tokens_eos_lens = self.hparams.fea_augment.replicate_labels(
                    tokens_eos_lens
                )

            # Decode token terms to words
            predicted_words = self.tokenizer(
                predicted_tokens, task="decode_from_list"
            )

            # Convert indices to words
            target_words = undo_padding(tokens, tokens_lens)
            target_words = self.tokenizer(target_words, task="decode_from_list")

            wer_stats = sb.utils.metric_stats.ErrorRateStats()
            wer_stats.append(ids=ids, predict=predicted_words, target=target_words)
            stats = wer_stats.summarize()
            wer = stats['WER']

        loss_seq = self.hparams.seq_cost(
            p_seq, tokens_eos, length=tokens_eos_lens
        )

        # Add ctc loss if necessary
        current_epoch = self.hparams.epoch_counter.current
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
            if if_main_process():
                with open(self.hparams.test_wer_file, "w") as w:
                    self.wer_metric.write_stats(w)
            return stage_stats["WER"]

    def fit_batch(self, batch):
        amp = AMPConfig.from_name(self.precision)
        should_step = (self.step % self.grad_accumulation_factor) == 0

        with self.no_sync(not should_step):
            if self.use_amp:
                with torch.autocast(
                    dtype=amp.dtype, device_type=torch.device(self.device).type,
                ):
                    outputs = self.compute_forward(batch, sb.Stage.TRAIN)
                    loss, wer = self.compute_objectives(
                        outputs, batch, sb.Stage.TRAIN
                    )
            else:
                outputs = self.compute_forward(batch, sb.Stage.TRAIN)
                loss, wer = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)

            scaled_loss = self.scaler.scale(
                loss / self.grad_accumulation_factor
            )
            self.check_loss_isfinite(scaled_loss)
            scaled_loss.backward()

        if should_step:
            self.optimizers_step()

        self.on_fit_batch_end(batch, outputs, loss, should_step)
        return loss.detach().cpu(), wer

    def fit(
            self,
            epoch_counter,
            train_set,
            valid_set=None,
            progressbar=None,
            train_loader_kwargs={},
            valid_loader_kwargs={},
    ):
        if self.test_only:
            return

        if not (
            isinstance(train_set, DataLoader)
            or isinstance(train_set, LoopedLoader)
        ):
            train_set = self.make_dataloader(
                train_set, stage=sb.Stage.TRAIN, **train_loader_kwargs
            )
        if valid_set is not None and not (
            isinstance(valid_set, DataLoader)
            or isinstance(valid_set, LoopedLoader)
        ):
            valid_set = self.make_dataloader(
                valid_set,
                stage=sb.Stage.VALID,
                ckpt_prefix=None,
                **valid_loader_kwargs,
            )

        self.on_fit_start()

        if progressbar is None:
            progressbar = not self.noprogressbar

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

    def evaluate(
        self,
        test_set,
        progressbar=None,
        test_loader_kwargs={},
    ):
        if progressbar is None:
            progressbar = not self.noprogressbar

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

