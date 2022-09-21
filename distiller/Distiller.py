from typing import Tuple
import torch
from torch.nn import Module
from torch import Tensor
import torch.nn as nn
from torch.nn import CrossEntropyLoss, KLDivLoss, CosineEmbeddingLoss
from transformers import PretrainedConfig
from transformers.models.marian.modeling_marian import MarianEncoder, MarianModel, MarianMTModel

"Heavily influenced from https://gist.github.com/remi-or/4814577c59f4f38fcc89729ce4ba21e6"

def create_student_model(teacher: Module,
                         n: int):
    """Create a student model from a teacher model.

    Args:
        teacher: The teacher model.
        n: The fraction of the teacher model to keep.

    Returns:
        A student model.
    """
    config = teacher.config.to_dict()
    config['num_hidden_layers'] //= n
    config = PretrainedConfig.from_dict(config)
    student_model = type(teacher)(config)
    student = student_model
    student.n = n
    init_student_weights_DFS(teacher, student)
    return student

def init_student_weights_DFS(teacher: Module,
                         student: Module):
    """Initialize the weights of a student model.

    Args:
        student: The student model.
        teacher: The teacher model.
    """
    if isinstance(teacher, MarianModel) or type(teacher).__name__.startswith('MarianNMT'):
        for t, s in zip(teacher.children(), student.children()):
            init_student_weights_DFS(t, s)
    elif isinstance(teacher, MarianEncoder):
        t_enc_layers = [l for l in next(teacher.children())]
        s_enc_layers = [l for l in next(student.children())]
        for i in range(len(s_enc_layers)):
            s_enc_layers[i].load_state_dict(t_enc_layers[i * student.n].state_dict())
    else:
        student.load_state_dict(teacher.state_dict())


class Distiller:
    """
    Distiller class for initializing a student model
    """

    def __init__(self,
                 teacher: MarianMTModel,
                 n: int,
                 temperature: float = 1.0,
                 loss_weights: list[float] = [1/3, 1/3, 1/3]
                 ):
        """
        Args:
            teacher: The teacher model.
            n: The fraction of the teacher model to keep.
            temperature: The temperature to use for distillation.
            loss_weights: The weights to use for the loss.
                loss_weights format:
                    [CE, KL, Cosine]
        """

        self.teacher = teacher
        self.student = create_student_model(teacher, n)
        self.n = n
        self._temperature = temperature
        self.loss_weights = loss_weights
        assert len(self.loss_weights) == 3, "loss_weights must be a list of length 3"
        assert torch.allclose(torch.tensor(1), torch.tensor(self.loss_weights).sum()), "loss_weights must sum to 1"

    @property
    def temperature(self):
        return self._temperature if self.training else 1

    @temperature.setter
    def temperature(self,
                    value: float):
        if temperature < 1:
            raise ValueError('Temperature must be greater than 1')
        self._temperature = value

    def get_logits_student(self,
                   input_ids: Tensor,
                   attention_mask: Tensor,
                   decoder_input_ids: Tensor,
                   decoder_attention_mask: Tensor,
                   **kwargs) -> Tensor:
        """
        Get the logits from the student model and the teacher model
        :param input_ids: The input ids
        :param attention_mask: The attention mask
        :param decoder_input_ids: The decoder input ids
        :param decoder_attention_mask: The decoder attention mask
        :param kwargs: Additional arguments
        :return: The student logits and the teacher logits
        """
        student_logits = self.student(input_ids=input_ids,
                                      attention_mask=attention_mask,
                                      decoder_input_ids=decoder_input_ids,
                                      decoder_attention_mask=decoder_attention_mask,
                                      **kwargs)[0]

        return student_logits

    def get_logits_teacher(self,
                    input_ids: Tensor,
                    attention_mask: Tensor,
                    decoder_input_ids: Tensor,
                    decoder_attention_mask: Tensor,
                    **kwargs) -> Tensor:
        """
        Get the logits from the student model and the teacher model
        :param input_ids: The input ids
        :param attention_mask: The attention mask
        :param decoder_input_ids: The decoder input ids
        :param decoder_attention_mask: The decoder attention mask
        :param kwargs: Additional arguments
        :return: The student logits and the teacher logits
        """
        with torch.no_grad():
            teacher_logits = self.teacher(input_ids=input_ids,
                                            attention_mask=attention_mask,
                                            decoder_input_ids=decoder_input_ids,
                                            decoder_attention_mask=decoder_attention_mask,
                                            **kwargs)[0]
        return teacher_logits

    def get_logits(self,
                  input_ids: Tensor,
                  attention_mask: Tensor,
                  decoder_input_ids: Tensor,
                  decoder_attention_mask: Tensor,
                  **kwargs) -> Tuple[Tensor, Tensor]:
          """
          Get the logits from the student model and the teacher model
          :param input_ids: The input ids
          :param attention_mask: The attention mask
          :param decoder_input_ids: The decoder input ids
          :param decoder_attention_mask: The decoder attention mask
          :param kwargs: Additional arguments
          :return: The student logits and the teacher logits
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
          return student_logits, teacher_logits

    def forward(self,
            input_ids: Tensor,
            attention_mask: Tensor,
            decoder_input_ids: Tensor,
            decoder_attention_mask: Tensor,
            labels: Tensor,
    ) -> Tensor:
        """
        Compute the loss between the student and teacher logits
        :param input_ids: The input ids
        :param attention_mask: The attention mask
        :param decoder_input_ids: The decoder input ids
        :param decoder_attention_mask: The decoder attention mask
        :param labels: The labels
        :return: The loss
        """
        student_logits, teacher_logits = self.get_logits(input_ids=input_ids,
                                                         attention_mask=attention_mask,
                                                         decoder_input_ids=decoder_input_ids,
                                                         decoder_attention_mask=decoder_attention_mask)
        return self.compute_loss(student_logits, teacher_logits, labels)

    def compute_loss(self,
                student_logits: Tensor,
                teacher_logits: Tensor,
                labels: Tensor) -> Tensor:
            """
            Compute the loss between the student and teacher logits
            :param student_logits: The student logits
            :param teacher_logits: The teacher logits
            :param labels: The labels
            :return: The loss
            """
            student_logits = (student_logits / self.temperature).softmax(1)
            teacher_logits = (teacher_logits / self.temperature).softmax(1)
            CE_loss = nn.CrossEntropyLoss()(student_logits, labels)
            KD_loss = nn.KLDivLoss()(student_logits.log(), teacher_logits)
            EM_loss = CosineEmbeddingLoss()(student_logits, teacher_logits, torch.ones_like(labels))
            loss = self.loss_weights[0] * CE_loss + self.loss_weights[1] * KD_loss + self.loss_weights[2] * EM_loss
            return loss