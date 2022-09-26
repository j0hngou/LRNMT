import sys
import pytorch_lightning as pl
sys.path.append('../')

from distiller_biling_teachers import DistillerBilingTeachers
from transformers import AutoModelForSeq2SeqLM
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from datamodules import MTDistillationDatamodule
from torch.nn import ModuleDict
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--teacher_path', nargs='+', type=str, required=True,
                    help='The path(or huggingface path) to the teacher model. If there are multiple, they should be separated by a space.')
parser.add_argument('--teacher_lang', nargs='+', type=str, required=True,
                    help='The language of the teacher model. If there are multiple, they should be separated by a space. \
                        For example, if the teacher_path is ["t5-small_en_ro", "t5-small_en_fr"], the teacher_lang should be ["en-ro", "en-fr"].')
parser.add_argument('--student_size', type=int, default=1, help='The size of the student model as a fraction of the teacher model.')
parser.add_argument('--temperature', type=float, default=1, help='The temperature to use for distillation.')
parser.add_argument('--loss_weights', nargs='+', type=float, default=[1/2, 1/2, 0], help='The weights to use for the loss. \
                    loss_weights format: [CE, KL, Cosine]')
parser.add_argument('--lr', type=float, default=2e-5, help='The learning rate.')
parser.add_argument('--weight_decay', type=float, default=0.01, help='The weight decay.')
parser.add_argument('--batch_size', type=int, default=8, help='The batch size.')
parser.add_argument('--max_epochs', type=int, default=10, help='The maximum number of epochs.')
parser.add_argument('--fp16', action='store_true', default=False, help='Whether to use mixed precision training.')
parser.add_argument('--wandb_project', type=str, default='distiller', help='The wandb project name.')
parser.add_argument('--val_check_interval', type=float, default=0.05, help='The validation check interval.')


args = parser.parse_args()

early_stop_callback = EarlyStopping(
    monitor='val_loss',
    min_delta=0.00,
    patience=3,
    verbose=False,
    mode='min'
)


wandb_logger = WandbLogger(project=args.wandb_project, entity="deeplearning2")

teachers = ModuleDict({lang: AutoModelForSeq2SeqLM.from_pretrained(path) for path, lang in zip(args.teacher_path, args.teacher_lang)})
batch_size = args.batch_size

dm = MTDistillationDatamodule(batch_size=batch_size)
dm.setup()

loss_weights = {'ce': args.loss_weights[0], 'kl': args.loss_weights[1], 'cosine': args.loss_weights[2]}

distiller = DistillerBilingTeachers(
    teachers=teachers,
    student_size=args.student_size,
    loss_weights=loss_weights,
    lr=args.lr,
    weight_decay=args.weight_decay,
)

trainer = pl.Trainer(
    gpus=1,
    max_epochs=args.max_epochs,
    logger=wandb_logger,
    callbacks=[early_stop_callback],
    val_check_interval=args.val_check_interval,
    precision=16 if args.fp16 else 32,
)

trainer.fit(distiller, dm)