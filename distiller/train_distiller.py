import sys
import pytorch_lightning as pl
sys.path.append('../')

from distiller_one_teacher import DistillerOneTeacher
from transformers import AutoModelForSeq2SeqLM
from datamodules import MTDistillationDatamodule
from transformers import AutoTokenizer
from pytorch_lightning.loggers import WandbLogger
import argparse
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

parser = argparse.ArgumentParser()
parser.add_argument('--teacher_path', type=str,
                    help='The path(or huggingface path) to the teacher model. If there are multiple, they should be separated by a space.',
                    default='t5-base')
parser.add_argument('--student_size', type=int, default=6,
                    help='The size of the student model as a fraction of the teacher model.')
parser.add_argument('--temperature', type=float, default=1, help='The temperature to use for distillation.')
parser.add_argument('--loss_weights', nargs='+', type=float, default=[1 / 2, 1 / 2, 0], help='The weights to use for the loss. \
                    loss_weights format: [CE, KL, Cosine]')
parser.add_argument('--lr', type=float, default=2e-5, help='The learning rate.')
parser.add_argument('--weight_decay', type=float, default=0.0, help='The weight decay.')
parser.add_argument('--batch_size', type=int, default=8, help='The batch size.')
parser.add_argument('--max_epochs', type=int, default=10, help='The maximum number of epochs.')
parser.add_argument('--fp16', action='store_true', default=False, help='Whether to use mixed precision training.')
parser.add_argument('--wandb_project', type=str, default='distiller', help='The wandb project name.')
parser.add_argument('--val_check_interval', type=float, default=0.05, help='The validation check interval.')
parser.add_argument('--seed', type=int, default=123, help='The seed to use.')
parser.add_argument('--random_initialized_student', action='store_true', help='Whether the student is random initialized. If not, the en-ro-t5-small model will be used.',
default=False)
parser.add_argument('--experiment_name', type=str, default='', help='The name of the experiment.')
parser.add_argument('--disable_dropout', action='store_true', help='Disables dropout in the student model.', default=False)

args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.teacher_path)

wandb_logger = WandbLogger(project=args.wandb_project, entity="deeplearning2",
                           name=f"{args.experiment_name}loss_weights_{args.loss_weights}_lr_{args.lr}_weight_decay_{args.weight_decay}_batch_size_{args.batch_size}_max_epochs_{args.max_epochs}")

pl.seed_everything(args.seed)
dm = MTDistillationDatamodule(batch_size=args.batch_size, group_pairs=False)
dm.setup()

distiller = DistillerOneTeacher(
    teacher=AutoModelForSeq2SeqLM.from_pretrained(args.teacher_path),
    n=args.student_size,
    temperature=args.temperature,
    loss_weights=args.loss_weights,
    weight_decay=args.weight_decay,
    disable_dropout=args.disable_dropout,
    precision=16 if args.fp16 else 32,
)

trainer = pl.Trainer(
    gpus=1,
    max_epochs=1,
    logger=wandb_logger,
    precision=16 if args.fp16 else 32,
    val_check_interval=args.val_check_interval,
)

trainer.fit(distiller, dm)
