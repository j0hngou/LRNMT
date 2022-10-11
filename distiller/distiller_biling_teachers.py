from torch.nn import CrossEntropyLoss, KLDivLoss, CosineEmbeddingLoss
from transformers import AutoModelForSeq2SeqLM
from torch.optim import Adam
from torch.nn import ModuleDict
from datasets import load_metric
from transformers import AutoTokenizer
from transformers.models.t5.modeling_t5 import T5Config
import math

import torch
import pytorch_lightning as pl


class DistillerBilingTeachers(pl.LightningModule):
    def __init__(self,
                 teachers: ModuleDict,
                 loss_weights: dict = {"ce": 1 / 2, "kl": 1 / 2},
                 lr: float = 2e-5,
                 weight_decay=0.01,
                 random_initialized_student: bool = False,
                 disable_dropout: bool = False,
                 precision: int = 32,
                 schedule=None,
                 decay_epochs=None,
                 warmup_steps=0,
                 init_path="din0s/t5-small-finetuned-en-to-ro",
                 ):
        super().__init__()
        self.save_hyperparameters(ignore=['teachers'])

        self.teachers = teachers
        for teacher in self.teachers.values():
            teacher.config.max_length = 256

        if random_initialized_student:
            self.student = AutoModelForSeq2SeqLM.from_config(config=T5Config.from_pretrained("t5-small"))
            self.student._init_weights(self.student)
            self.student.lm_head.reset_parameters()
        else:
            self.student = AutoModelForSeq2SeqLM.from_pretrained(init_path)

        if disable_dropout:
            self._disable_dropout()
        self.student.config.max_length = 256

        self.tokenizer = AutoTokenizer.from_pretrained("t5-small")

        self.ce_loss = CrossEntropyLoss(ignore_index=self.student.config.pad_token_id)
        self.kl_loss = KLDivLoss(reduction='batchmean')
        self.cosine_loss = CosineEmbeddingLoss()

        self.sacrebleu = load_metric("sacrebleu")

        assert len(loss_weights) == 3, "loss_weights must be a list of length 3"

    def _disable_dropout(self):
        self.student.encoder.dropout.p = 0.0
        self.student.decoder.dropout.p = 0.0
        for i in range(self.student.config.num_layers):
            for j in range(len(self.student.encoder.block[i].layer)):
                self.student.encoder.block[i].layer[j].dropout.p = 0.0
                if j == 1:
                    self.student.encoder.block[i].layer[j].DenseReluDense.dropout.p = 0.0

        for i in range(self.student.config.num_layers):
            for j in range(len(self.student.decoder.block[i].layer)):
                self.student.decoder.block[i].layer[j].dropout.p = 0.0
                if j == 2:
                    self.student.decoder.block[i].layer[j].DenseReluDense.dropout.p = 0.0

    def get_logits_student(self,
                           batch: dict,
                           ) -> dict:
        """
        Get the logits from the student model and the teacher model
        Args:
            batch: The batch to get the logits from
        Returns:
            The student logits
        """

        # TODO: [NMTDL4NLP-25] use this for parallel pass for all pairs, code in training step need to change
        # student_logits = self.student(input_ids=torch.stack([pair["input_ids"] for pair in batch.values()]),
        #                               attention_mask=torch.stack([pair["attention_mask"] for pair in batch.values()]),
        #                               decoder_input_ids=torch.stack([pair["decoder_input_ids"] for pair in batch.values()]),
        #                               decoder_attention_mask=torch.stack([pair["decoder_attention_mask"] for pair in batch.values()]),
        #                               **kwargs).logits

        logits = {}
        for pair in batch.keys():
            logits[pair] = self.student(input_ids=batch[pair]["input_ids"],
                                        attention_mask=batch[pair]["attention_mask"],
                                        labels=batch[pair]["decoder_input_ids"],
                                        ).logits

        return logits

    def get_logits_teacher(self,
                           batch: dict, ) -> dict:
        """
        Get the logits from the student model and the teacher model
        Args:
            batch: The batch to get the logits from
        Returns:
            The teacher logits
        """

        logits = {}

        for pair in batch.keys():
            with torch.no_grad():
                self.teachers[pair].eval()
                logits[pair] = self.teachers[pair](input_ids=batch[pair]["input_ids"],
                                                   attention_mask=batch[pair]["attention_mask"],
                                                   labels=batch[pair]["decoder_input_ids"],
                                                   ).logits

        return logits

    def forward(self,
                batch: dict,
                mode: str) -> dict:
        """
        Forward pass through the student and teacher model to get their logits for each language pair.
        Args:
            batch: The batch to pass through the student model and the teacher model
        Returns:
            The student logits

        """
        student_logits = self.get_logits_student(batch)
        teacher_logits = self.get_logits_teacher(batch)

        metrics = self._compute_ce_kl(student_logits, teacher_logits, batch)
        if mode == "val" or mode == "test":
            self._compute_bleu(batch)

        return metrics

    def training_step(self,
                      batch: dict,
                      batch_idx: int, ) -> dict:
        if self.hparams.schedule == "linear":
            self._update_loss_weights()
        metrics = self.forward(batch, mode="train")

        self.log('train_loss', metrics["loss"])
        self.log('train_ce_loss', metrics["ce_loss"])
        self.log('train_kl_loss', metrics["kl_loss"])

        return metrics

    def validation_step(self,
                        batch: dict,
                        batch_idx: int, ) -> dict:

        metrics = self.forward(batch, mode="val")
        return metrics

    def validation_epoch_end(self, outputs: dict) -> dict:
        outputs = self._test_eval_epoch_end(outputs, mode="val")
        return outputs

    def test_step(self,
                  batch: dict,
                  batch_idx: int, ) -> dict:

        metrics = self.forward(batch, mode="test")
        return metrics

    def test_epoch_end(self, outputs: dict) -> dict:
        outputs = self._test_eval_epoch_end(outputs, "test")
        return outputs

    def _test_eval_epoch_end(self, outputs: dict, mode: str) -> dict:
        bleu_score = self.sacrebleu.compute()["score"]
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_ce_loss = torch.stack([x['ce_loss'] for x in outputs]).mean()
        avg_kl_loss = torch.stack([x['kl_loss'] for x in outputs]).mean()
        self.log(f'{mode}_bleu', bleu_score)
        self.log(f'{mode}_loss', avg_loss)
        self.log(f'{mode}_ce_loss', avg_ce_loss)
        self.log(f'{mode}_kl_loss', avg_kl_loss)
        return {f"{mode}_loss": avg_loss, f"{mode}_bleu": bleu_score}

    def _compute_ce_kl(self, student_logits: dict, teacher_logits: dict, batch: dict) -> dict:
        # Cross entropy loss
        ce_loss = 0
        total_samples = 0
        for pair in student_logits.keys():
            num_samples = batch[pair]["decoder_input_ids"].shape[0]
            total_samples += num_samples
            ce_loss += num_samples * self.ce_loss(student_logits[pair].permute(0, 2, 1),
                                                  batch[pair]["decoder_input_ids"])

        ce_loss /= total_samples

        # KL divergence loss
        kl_loss = 0
        total_samples = 0
        for pair in teacher_logits.keys():
            num_samples = batch[pair]["decoder_input_ids"].shape[0]
            total_samples += num_samples
            pad_token_id = self.tokenizer.pad_token_id
            student_logits[pair][batch[pair]["decoder_input_ids"] == pad_token_id] = -65504 if self.hparams.precision == 16 else -1e9
            teacher_logits[pair][batch[pair]["decoder_input_ids"] == pad_token_id] = -65504 if self.hparams.precision == 16 else -1e9
            kl_loss += num_samples * self.kl_loss(torch.log_softmax(student_logits[pair], dim=-1),
                                                  torch.softmax(teacher_logits[pair], dim=-1))
        kl_loss /= total_samples

        # Cosine loss
        # cosine_loss = self.loss_weights[2] * self.cosine_loss(student_logits, teacher_logits, torch.ones_like(student_logits[:, 0]))

        loss = self.hparams.loss_weights["ce"] * ce_loss + self.hparams.loss_weights["kl"] * kl_loss  # + cosine_loss

        return {"loss": loss, "ce_loss": ce_loss, "kl_loss": kl_loss}

    def _compute_bleu(self, batch: dict):
        for pair in batch.keys():
            prediction_ids = self.student.generate(batch[pair]["input_ids"], num_beams=5)
            prediction = self.tokenizer.batch_decode(prediction_ids, skip_special_tokens=True)
            target = self.tokenizer.batch_decode(batch[pair]["decoder_input_ids"], skip_special_tokens=True)
            target = [[t] for t in target]
            self.sacrebleu.add_batch(predictions=prediction, references=target)

    def _update_loss_weights(self):
        if self.global_step < self.hparams.warmup_steps:
            self.hparams.loss_weights["kl"] = 1.0
            self.hparams.loss_weights["ce"] = 0.0
        else:
            if self.hparams.schedule == "linear":
                # Linearly decrease KL from 1 to 0.2
                self.hparams.loss_weights["kl"] = max(0.2, 1 - self.current_epoch / self.hparams.decay_epochs)
                self.hparams.loss_weights["ce"] = 1 - self.hparams.loss_weights["kl"]
            elif self.hparams.schedule == "cosine":
                # Cosine decay
                self.hparams.loss_weights["kl"] = 0.5 * (1 + math.cos(math.pi * self.current_epoch / self.hparams.decay_epochs))
                self.hparams.loss_weights["ce"] = 1 - self.hparams.loss_weights["kl"]
            else:
                raise NotImplementedError(f"Schedule {self.hparams.schedule} not implemented")

    def configure_optimizers(self):
        optimizer = Adam(self.student.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        return optimizer


class DistillerEnItTeachers(DistillerBilingTeachers):
    def __init__(self,
                 teachers: ModuleDict,
                 loss_weights: dict = {"ce": 1 / 2, "kl": 1 / 2},
                 lr: float = 2e-5,
                 weight_decay=0.01,
                 random_initialized_student: bool = False,
                 disable_dropout: bool = False,
                 precision: int = 32,
                 init_path="din0s/t5-small-finetuned-en-to-ro",
                 **kwargs):
        super().__init__(teachers=teachers, loss_weights=loss_weights, lr=lr, weight_decay=weight_decay,
                         random_initialized_student=random_initialized_student, disable_dropout=disable_dropout,
                         precision=precision, init_path=init_path, **kwargs)

        self.pair = "en-it"

    def get_logits_student(self,
                           batch: dict,
                           ):
        logits = self.student(input_ids=batch[self.pair]["input_ids"],
                              attention_mask=batch[self.pair]["attention_mask"],
                              labels=batch[self.pair]["decoder_input_ids"],
                              ).logits

        return logits

    def get_logits_teacher(self,
                           batch: dict, ) -> dict:
        logits = {}

        for pair in self.teachers.keys():
            with torch.no_grad():
                self.teachers[pair].eval()
                logits[pair] = self.teachers[pair](input_ids=batch[self.pair]["input_ids"],
                                                   attention_mask=batch[self.pair]["attention_mask"],
                                                   labels=batch[self.pair]["decoder_input_ids"],
                                                   ).logits

        return logits

    def _compute_ce_kl(self, student_logits: torch.Tensor, teacher_logits: dict, batch: dict) -> dict:
        # Cross entropy loss
        ce_loss = self.ce_loss(student_logits.permute(0, 2, 1), batch[self.pair]["decoder_input_ids"])

        # KL divergence loss
        kl_loss = 0
        pad_token_id = self.tokenizer.pad_token_id
        student_logits[batch[self.pair]["decoder_input_ids"] == pad_token_id] = -65504 if self.hparams.precision == 16 else -1e9
        for pair in teacher_logits.keys():
            teacher_logits[pair][batch[self.pair]["decoder_input_ids"] == pad_token_id] = -65504 if self.hparams.precision == 16 else -1e9
            kl_loss += self.kl_loss(torch.log_softmax(student_logits, dim=-1),
                                                  torch.softmax(teacher_logits[pair], dim=-1))
        kl_loss /= len(teacher_logits.keys())

        loss = self.hparams.loss_weights["ce"] * ce_loss + self.hparams.loss_weights["kl"] * kl_loss  # + cosine_loss

        return {"loss": loss, "ce_loss": ce_loss, "kl_loss": kl_loss}
