import sys
import pytorch_lightning as pl
sys.path.append('../')

from distiller_biling_teachers import DistillerBilingTeachers
from transformers import AutoModelForSeq2SeqLM
from pytorch_lightning.loggers import WandbLogger
from datamodules import MTDistillationDatamodule
from torch.nn import ModuleDict


wandb_logger = WandbLogger(project="distiller", entity="deeplearning2")

teachers = ModuleDict({"en-ro": AutoModelForSeq2SeqLM.from_pretrained("t5-small"),
                       "en-fr": AutoModelForSeq2SeqLM.from_pretrained("t5-small"),
                       "en-de": AutoModelForSeq2SeqLM.from_pretrained("t5-small")})
batch_size = 10

dm = MTDistillationDatamodule(batch_size=batch_size)
dm.setup()

distiller = DistillerBilingTeachers(
    teachers=teachers,
    loss_weights=[1/2, 1/2, 0],
)

trainer = pl.Trainer(
    gpus=1,
    max_epochs=1,
    logger=wandb_logger,
    # precision=16, # lisa fp16
    # amp_backend="apex", # lisa fp16
    # amp_level="O2", # lisa fp16
)

trainer.fit(distiller, dm)