from torch.nn import Module
from torch import Tensor
from torch.nn import CrossEntropyLoss, KLDivLoss, CosineEmbeddingLoss
from transformers import PretrainedConfig
from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration
from torch.optim import Adam
from datasets import load_metric
from transformers import AutoTokenizer
from transformers.models.t5.modeling_t5 import T5Config

import torch
import pytorch_lightning as pl


class DistillerOneTeacher(pl.LightningModule):
    def __init__(self,
                 teacher: T5ForConditionalGeneration,
                 n: int,
                 loss_weights: list[float] = [1/2, 1/2, 0],
                 lr: float = 2e-5,
                 weight_decay=0.01,
                 disable_dropout=True,
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
        self.teacher.config.max_length = 256
        self.student = self._create_student_model()
        self.student.config.max_length = 256
        self.loss_weights = loss_weights

        self.ce_loss = CrossEntropyLoss(ignore_index=self.student.config.pad_token_id)
        self.kl_loss = KLDivLoss(reduction='batchmean')
        self.cosine_loss = CosineEmbeddingLoss()
        if disable_dropout:
            self._disable_dropout()

        assert len(loss_weights) == 3, "loss_weights must be a list of length 3"
        self.sacrebleu = load_metric("sacrebleu")
        self.tokenizer = AutoTokenizer.from_pretrained("t5-small")



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
                mode: str = 'train',
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
        student_logits = self.get_logits_student(input_ids=input_ids,
                                                    attention_mask=attention_mask,
                                                    decoder_input_ids=decoder_input_ids,
                                                    decoder_attention_mask=decoder_attention_mask,
                                                    **kwargs)
    
        teacher_logits = self.get_logits_teacher(input_ids=input_ids,
                                                    attention_mask=attention_mask,
                                                    decoder_input_ids=decoder_input_ids,
                                                    decoder_attention_mask=decoder_attention_mask,
                                                    **kwargs)

        metrics = self._compute_ce_kl(student_logits, teacher_logits, 
                                      decoder_input_ids, decoder_attention_mask)

        if mode == 'val' or mode == 'test':
            self._compute_bleu(input_ids, decoder_input_ids)
        return metrics

    def _compute_ce_kl(self, student_logits, teacher_logits, decoder_input_ids, decoder_attention_mask):
        ce_loss = self.ce_loss(student_logits.permute(0, 2, 1), decoder_input_ids)
        student_logits[decoder_attention_mask == 0] = -1e9
        teacher_logits[decoder_attention_mask == 0] = -1e9
        kl_loss = self.kl_loss(torch.nn.functional.log_softmax(student_logits, dim=-1),
                               torch.nn.functional.softmax(teacher_logits, dim=-1))
        loss = self.loss_weights[0] * ce_loss + self.loss_weights[1] * kl_loss
        return {'loss': loss, 'ce_loss': ce_loss, 'kl_loss': kl_loss}

    def _compute_bleu(self, input_ids, decoder_input_ids):
        prediction_ids = self.student.generate(input_ids=input_ids,
                                               num_beams=5)
        prediction = self.tokenizer.batch_decode(prediction_ids, skip_special_tokens=True)
        target = self.tokenizer.batch_decode(decoder_input_ids, skip_special_tokens=True)
        target = [[t] for t in target]
        self.sacrebleu.add_batch(predictions=prediction, references=target)


    def training_step(self,
                        batch: dict,
                        batch_idx: int,) -> Tensor:

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        decoder_input_ids = batch["decoder_input_ids"]
        decoder_attention_mask = batch["decoder_attention_mask"]
        metrics = self.forward(input_ids, attention_mask, decoder_input_ids, decoder_attention_mask)

        self.log('train_loss', metrics['loss'], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_ce_loss', metrics['ce_loss'], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_kl_loss', metrics['kl_loss'], on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return metrics

    def validation_step(self,
                        batch: dict,
                        batch_idx: int,) -> Tensor:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        decoder_input_ids = batch["decoder_input_ids"]
        decoder_attention_mask = batch["decoder_attention_mask"]
        metrics = self.forward(input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, mode='val')
        
        return metrics

    def validation_epoch_end(self, outputs: dict):
        outputs = self._test_eval_epoch_end(outputs, mode='val')
        return outputs

    def test_step(self,
                        batch: dict,
                        batch_idx: int,) -> Tensor:

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        decoder_input_ids = batch["decoder_input_ids"]
        decoder_attention_mask = batch["decoder_attention_mask"]
        metrics = self.forward(input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, mode='test')

        return metrics

    def test_epoch_end(self, outputs: dict):
        outputs = self._test_eval_epoch_end(outputs, mode='test')
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
