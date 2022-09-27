from typing import Tuple
from torch.nn import CrossEntropyLoss, KLDivLoss, CosineEmbeddingLoss
from transformers import AutoModelForSeq2SeqLM
from torch.optim import Adam
from torch.nn import ModuleDict
from datasets import load_metric
from transformers import AutoTokenizer

import torch
import pytorch_lightning as pl


class DistillerBilingTeachers(pl.LightningModule):
    def __init__(self,
                 teachers: ModuleDict,
                 loss_weights: dict = {"ce": 1 / 2, "kl": 1 / 2},
                 temperature=1,
                 lr: float = 2e-5,
                 weight_decay=0.01,
                 **kwargs):
        """
        Args:
            teacher: The teacher model.
            n: The fraction of the teacher model to keep.
            temperature: The temperature to use for distillation.
            loss_weights: The weights to use for the loss.
            lr: The learning rate
            kwargs: Additional arguments
        """
        super().__init__()
        self.save_hyperparameters(ignore=['teachers'])

        self.teachers = teachers
        for teacher in self.teachers.values():
            teacher.config.max_length = 256
        self.student = AutoModelForSeq2SeqLM.from_pretrained("din0s/t5-small-finetuned-en-to-ro")
        self.student.tokenizer = AutoTokenizer.from_pretrained("din0s/t5-small-finetuned-en-to-ro")
        self.student.config.max_length = 256

        self.ce_loss = CrossEntropyLoss()
        self.kl_loss = KLDivLoss(reduction='batchmean')
        self.cosine_loss = CosineEmbeddingLoss()

        self.sacrebleu = load_metric("sacrebleu")

        assert len(loss_weights) == 3, "loss_weights must be a list of length 3"

    def get_logits_student(self,
                           batch: dict,
                           **kwargs) -> dict:
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
                                        decoder_input_ids=batch[pair]["decoder_input_ids"],
                                        decoder_attention_mask=batch[pair]["decoder_attention_mask"],
                                        **kwargs).logits

        return logits

    def get_logits_teacher(self,
                           batch: dict,
                           **kwargs) -> dict:
        """
        Get the logits from the student model and the teacher model
        Args:
            batch: The batch to get the logits from
            kwargs: Additional arguments
        Returns:
            The teacher logits
        """

        logits = {}
        for pair in batch.keys():
            with torch.no_grad():
                self.teachers[pair].eval()
                logits[pair] = self.teachers[pair](input_ids=batch[pair]["input_ids"],
                                                   attention_mask=batch[pair]["attention_mask"],
                                                   decoder_input_ids=batch[pair]["decoder_input_ids"],
                                                   decoder_attention_mask=batch[pair]["decoder_attention_mask"],
                                                   **kwargs).logits

        return logits

    def forward(self,
                batch: dict) -> Tuple[dict, dict]:
        """
        Forward pass through the student and teacher model to get their logits for each language pair.
        Args:
            batch: The batch to pass through the student model and the teacher model
        Returns:
            The student logits

        """
        student_logits = self.get_logits_student(batch)
        teacher_logits = self.get_logits_teacher(batch)

        return student_logits, teacher_logits

    def training_step(self,
                      batch: dict,
                      batch_idx: int, ) -> dict:

        student_logits, teacher_logits = self.forward(batch)

        metrics = self._compute_metrics(student_logits, teacher_logits, batch, mode="train")

        self.log('train_loss', metrics["loss"])
        self.log('train_ce_loss', metrics["ce_loss"])
        self.log('train_kl_loss', metrics["kl_loss"])

        return metrics

    def validation_step(self,
                        batch: dict,
                        batch_idx: int, ) -> dict:

        student_logits, teacher_logits = self.forward(batch)
        metrics = self._compute_metrics(student_logits, teacher_logits, batch, mode="valid")

        self.log('valid_loss', metrics["loss"])
        self.log('valid_ce_loss', metrics["ce_loss"])
        self.log('valid_kl_loss', metrics["kl_loss"])

        return metrics

    def validation_epoch_end(self, outputs: list) -> dict:
        bleu_score = self.sacrebleu.compute()
        self.log('val_bleu', bleu_score)
        return {'val_bleu': bleu_score}

    def test_step(self,
                        batch: dict,
                        batch_idx: int, ) -> dict:

        student_logits, teacher_logits = self.forward(batch)
        metrics = self._compute_metrics(student_logits, teacher_logits, batch, mode="test")

        self.log('test_loss', metrics["loss"])
        self.log('test_ce_loss', metrics["ce_loss"])
        self.log('test_kl_loss', metrics["kl_loss"])

        return metrics

    def test_epoch_end(self, outputs: list) -> dict:
        bleu_score = self.sacrebleu.compute()
        self.log('test_bleu', bleu_score)
        return {'test_bleu': bleu_score}

    def _compute_metrics(self, student_logits: dict, teacher_logits: dict, batch: dict, mode: str) -> dict:
        """
        Currently only calculates SacreBLEU
        """
        tokenizer = self.student.tokenizer

        # Cross entropy loss and unormalized perplexities
        ce_loss = 0
        perplexities = {}
        for pair in student_logits.keys():
            ce_loss += self.ce_loss(student_logits[pair].permute(0, 2, 1), batch[pair]["decoder_input_ids"])
            perplexities[pair] = torch.exp(
                -self.ce_loss(teacher_logits[pair].permute(0, 2, 1), batch[pair]["decoder_input_ids"]))

        ce_loss /= len(student_logits.keys())
        ce_loss *= self.hparams.loss_weights["ce"]

        # KL divergence loss
        kl_loss = 0
        for pair in teacher_logits.keys():
            perplexities[pair] = perplexities[pair] / sum(perplexities.values())
            pad_token_id = tokenizer.pad_token_id
            student_logits[pair][batch[pair]["decoder_input_ids"] == pad_token_id] = -float("inf")
            teacher_logits[pair][batch[pair]["decoder_input_ids"] == pad_token_id] = -float("inf")
            kl_loss += perplexities[pair] * self.kl_loss(torch.log_softmax(student_logits[pair], dim=-1),
                                                         torch.softmax(teacher_logits[pair], dim=-1))

            kl_loss += perplexities[pair] * 
        kl_loss /= len(teacher_logits.keys())
        kl_loss *= self.hparams.loss_weights["kl"]

        # Cosine loss
        # cosine_loss = self.loss_weights[2] * self.cosine_loss(student_logits, teacher_logits, torch.ones_like(student_logits[:, 0]))

        loss = ce_loss + kl_loss  # + cosine_loss

        if mode == "valid" or mode == "test":
            for pair in student_logits.keys():
                _t = batch[pair]['decoder_input_ids']
                _t[_t == -100] = tokenizer.pad_token_id
                logits_pair = student_logits[pair].argmax(dim=-1)
                decoded_preds = tokenizer.batch_decode(logits_pair, skip_special_tokens=True)
                decoded_preds = [pred.strip() for pred in decoded_preds]
                decoded_labels = tokenizer.batch_decode(_t, skip_special_tokens=True)
                decoded_labels = [label.strip() for label in decoded_labels]
                decoded_labels = [[x] for x in decoded_labels]
                self.sacrebleu.add_batch(predictions=decoded_preds, references=decoded_labels)

        return {"loss": loss, "ce_loss": ce_loss, "kl_loss": kl_loss}

    def configure_optimizers(self):
        optimizer = Adam(self.student.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        return optimizer
