from torch import Tensor
from torch.nn import CrossEntropyLoss, KLDivLoss, CosineEmbeddingLoss
from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration
from torch.optim import Adam
from torch.nn import ModuleDict

import torch
import pytorch_lightning as pl


class DistillerBilingTeachers(pl.LightningModule):
    def __init__(self,
                 teachers: ModuleDict,
                 loss_weights: list[float] = [1/3, 1/3, 1/3],
                 lr: float = 2e-5,
                 weight_decay=0.01,
                 **kwargs):
        """
        Args:
            teacher: The teacher model.
            n: The fraction of the teacher model to keep.
            temperature: The temperature to use for distillation.
            loss_weights: The weights to use for the loss.
                loss_weights format:
                    [CE, KL, Cosine]
            lr: The learning rate
            kwargs: Additional arguments
        """
        super().__init__()
        self.save_hyperparameters(ignore=['teachers'])

        self.teachers = teachers
        # TODO: copy the weights from the teacher with the closest language
        self.student = T5ForConditionalGeneration.from_pretrained("t5-small")

        self.ce_loss = CrossEntropyLoss()
        self.kl_loss = KLDivLoss(reduction='batchmean')
        self.cosine_loss = CosineEmbeddingLoss()

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
                input_ids: Tensor,
                attention_mask: Tensor,
                decoder_input_ids: Tensor,
                decoder_attention_mask: Tensor,
                **kwargs) -> Tensor:
        """
        Forward pass through the student model
        Args:
            input_ids: The input ids
            attention_mask: The attention mask
            decoder_input_ids: The decoder input ids
            decoder_attention_mask: The decoder attention mask
            kwargs: Additional arguments
        Returns:
            The student logits

        """
        return self.student(input_ids=input_ids,
                            attention_mask=attention_mask,
                            decoder_input_ids=decoder_input_ids,
                            decoder_attention_mask=decoder_attention_mask,
                            **kwargs).logits

    def training_step(self,
                        batch: dict,
                        batch_idx: int,) -> Tensor:

        student_logits = self.get_logits_student(batch)
        teacher_logits = self.get_logits_teacher(batch)

        # Cross entropy loss
        ce_loss = 0
        perplexities = []
        for pair in teacher_logits.keys():
            teacher_loss = self.ce_loss(student_logits[pair].permute(0, 2, 1), batch[pair]["decoder_input_ids"])
            ce_loss += teacher_loss
            perplexities.append(self._calculate_perplexity(teacher_loss))
        ce_loss /= len(teacher_logits.keys())
        ce_loss *= self.hparams.loss_weights[0]

        # Normalize perplexities to get the teacher weights
        teacher_weights = [perplexity / sum(perplexities) for perplexity in perplexities]
        teacher_weights = torch.tensor(teacher_weights)
        # KL divergence loss
        kl_loss = 0
        for i, pair in enumerate(teacher_logits.keys()):
            kl_loss += teacher_weights[i]*self.kl_loss(torch.softmax(student_logits[pair], dim=-1),
                                                       torch.softmax(teacher_logits[pair], dim=-1))
        kl_loss /= len(teacher_logits.keys())
        kl_loss *= self.hparams.loss_weights[1]

        # Cosine loss
        # cosine_loss = self.loss_weights[2] * self.cosine_loss(student_logits, teacher_logits, torch.ones_like(student_logits[:, 0]))

        loss = ce_loss + kl_loss# + cosine_loss

        self.log("ce_loss", ce_loss)
        self.log("kl_loss", kl_loss)
        self.log("train_loss", loss)

        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.student.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        return optimizer

    def _calculate_perplexity(self, loss):
        return torch.exp(loss)