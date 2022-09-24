from torch.nn import Module
from torch import Tensor
from torch.nn import CrossEntropyLoss, KLDivLoss, CosineEmbeddingLoss
from transformers import PretrainedConfig
from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration
from torch.optim import Adam

import torch
import pytorch_lightning as pl


class DistillerOneTeacher(pl.LightningModule):
    def __init__(self,
                 teacher: T5ForConditionalGeneration,
                 n: int,
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
        self.save_hyperparameters(ignore=['teacher'])

        self.teacher = teacher
        self.student = self._create_student_model()

        self.ce_loss = CrossEntropyLoss()
        self.kl_loss = KLDivLoss(reduction='batchmean')
        self.cosine_loss = CosineEmbeddingLoss()

        assert len(loss_weights) == 3, "loss_weights must be a list of length 3"

    def get_logits_student(self,
                     input_ids: Tensor,
                        attention_mask: Tensor,
                        decoder_input_ids: Tensor,
                        decoder_attention_mask: Tensor,
                        **kwargs) -> Tensor:
        """
        Get the logits from the student model and the teacher model
        Args:
            input_ids: The input ids
            attention_mask: The attention mask
            decoder_input_ids: The decoder input ids
            decoder_attention_mask: The decoder attention mask
            kwargs: Additional arguments
        Returns:
            The student logits
        """
        student_logits = self.student(input_ids=input_ids,
                                      attention_mask=attention_mask,
                                      decoder_input_ids=decoder_input_ids,
                                      decoder_attention_mask=decoder_attention_mask,
                                      **kwargs).logits

        return student_logits

    def get_logits_teacher(self,
                     input_ids: Tensor,
                        attention_mask: Tensor,
                        decoder_input_ids: Tensor,
                        decoder_attention_mask: Tensor,
                        **kwargs) -> Tensor:
        """
        Get the logits from the student model and the teacher model
        Args:
            input_ids: The input ids
            attention_mask: The attention mask
            decoder_input_ids: The decoder input ids
            decoder_attention_mask: The decoder attention mask
            kwargs: Additional arguments
        Returns:
            The teacher logits
        """
        self.teacher.eval()
        with torch.no_grad():
            teacher_logits = self.teacher(input_ids=input_ids,
                                        attention_mask=attention_mask,
                                        decoder_input_ids=decoder_input_ids,
                                        decoder_attention_mask=decoder_attention_mask,
                                        **kwargs).logits

        return teacher_logits

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

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        decoder_input_ids = batch["decoder_input_ids"]
        decoder_attention_mask = batch["decoder_attention_mask"]
        # labels = batch["labels"]

        student_logits = self.get_logits_student(input_ids,
                                                 attention_mask,
                                                 decoder_input_ids,
                                                 decoder_attention_mask)
        teacher_logits = self.get_logits_teacher(input_ids,
                                                 attention_mask,
                                                 decoder_input_ids,
                                                 decoder_attention_mask)

        # Cross entropy loss
        ce_loss = self.hparams.loss_weights[0] * self.ce_loss(student_logits.permute(0, 2, 1), decoder_input_ids)

        # KL divergence loss
        kl_loss = self.hparams.loss_weights[1] * self.kl_loss(student_logits.log_softmax(-1), teacher_logits.softmax(dim=-1))

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

    def _create_student_model(self):
        """Create a student model from a teacher model.

        Args:
            teacher: The teacher model.
            n: The fraction of the teacher model to keep.

        Returns:
            A student model.
        """
        config = self.teacher.config.to_dict()
        config['num_layers'] //= self.hparams.n
        config['num_decoder_layers'] //= self.hparams.n
        config = PretrainedConfig.from_dict(config)
        student_model = type(self.teacher)(config)
        student = student_model
        student.n = self.hparams.n
        self._init_student_weights(self.teacher, student)
        return student

    @staticmethod
    def _init_student_weights(teacher: Module,
                              student: Module):
        """Initialize the weights of a student model.

        Args:
            student: The student model.
            teacher: The teacher model.
        """
        student.shared.weight.data = teacher.shared.weight.data
        student.encoder.final_layer_norm.weight.data = teacher.encoder.final_layer_norm.weight.data
        student.decoder.final_layer_norm.weight.data = teacher.decoder.final_layer_norm.weight.data
        # Encoder
        for i in range(student.config.num_layers):
            student.encoder.block[i].load_state_dict(teacher.encoder.block[i * student.n].state_dict())
        # Decoder
        for i in range(student.config.num_decoder_layers):
            student.decoder.block[i].load_state_dict(teacher.decoder.block[i * student.n].state_dict())
