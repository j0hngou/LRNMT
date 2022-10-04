import sys
import pytorch_lightning as pl
sys.path.append('../')

from distiller_one_teacher import DistillerOneTeacher
from transformers import AutoModelForSeq2SeqLM
from datamodules import MTDistillationDatamodule
from transformers import AutoTokenizer
from pytorch_lightning.loggers import WandbLogger

wandb_logger = WandbLogger(project="distiller", entity="deeplearning2")

teacher_checkpoint = "t5-base"

# TODO: parser
# parser = argparse.ArgumentParser()

tokenizer = AutoTokenizer.from_pretrained(teacher_checkpoint)

batch_size = 2

dm = MTDistillationDatamodule(batch_size=batch_size, group_pairs=False)
dm.setup()

distiller = DistillerOneTeacher(
    teacher=AutoModelForSeq2SeqLM.from_pretrained(teacher_checkpoint),
    n=6,
    temperature=1,
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