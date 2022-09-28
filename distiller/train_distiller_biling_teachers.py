import sys
import pytorch_lightning as pl
import argparse

sys.path.append('../')

from distiller_biling_teachers import DistillerBilingTeachers
from transformers import AutoModelForSeq2SeqLM
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from datamodules import MTDistillationDatamodule
from torch.nn import ModuleDict


parser = argparse.ArgumentParser()
parser.add_argument('--teacher_path', nargs='+', type=str,
                    help='The path(or huggingface path) to the teacher model. If there are multiple, they should be separated by a space.',
                    default=["din0s/t5-small-finetuned-en-to-de", "din0s/t5-small-finetuned-en-to-fr",
                             "din0s/t5-small-finetuned-en-to-ro"])
parser.add_argument('--teacher_lang', nargs='+', type=str,
                    help='The language of the teacher model. If there are multiple, they should be separated by a space. \
                        For example, if the teacher_path is ["t5-small_en_ro", "t5-small_en_fr"], the teacher_lang should be ["en-ro", "en-fr"].',
                    default=["en-de", "en-fr", "en-ro"])
parser.add_argument('--student_size', type=int, default=1,
                    help='The size of the student model as a fraction of the teacher model.')
parser.add_argument('--temperature', type=float, default=1, help='The temperature to use for distillation.')
parser.add_argument('--loss_weights', nargs='+', type=float, default=[1 / 2, 1 / 2, 0], help='The weights to use for the loss. \
                    loss_weights format: [CE, KL, Cosine]')
parser.add_argument('--lr', type=float, default=2e-5, help='The learning rate.')
parser.add_argument('--weight_decay', type=float, default=0.01, help='The weight decay.')
parser.add_argument('--batch_size', type=int, default=8, help='The batch size.')
parser.add_argument('--max_epochs', type=int, default=10, help='The maximum number of epochs.')
parser.add_argument('--fp16', action='store_true', default=False, help='Whether to use mixed precision training.')
parser.add_argument('--wandb_project', type=str, default='distiller', help='The wandb project name.')
parser.add_argument('--val_check_interval', type=float, default=0.05, help='The validation check interval.')
parser.add_argument('--seed', type=int, default=123, help='The seed to use.')
parser.add_argument('--random_initialized_student', action='store_true', help='Whether the student is random initialized. If not, the en-ro-t5-small model will be used.',
default=False)
parser.add_argument('--experiment_name', type=str, default='', help='The name of the experiment.')

args = parser.parse_args()

pl.seed_everything(args.seed)
dm = MTDistillationDatamodule(batch_size=args.batch_size)
dm.setup()

early_stop_callback = EarlyStopping(
    monitor='val_bleu',
    min_delta=0.00,
    patience=5,
    verbose=False,
    mode='max'
)

wandb_logger = WandbLogger(project=args.wandb_project, entity="deeplearning2",
                           name=f"{args.experiment_name}loss_weights_{args.loss_weights}_lr_{args.lr}_weight_decay_{args.weight_decay}_batch_size_{args.batch_size}_max_epochs_{args.max_epochs}")

teachers = ModuleDict(
    {lang: AutoModelForSeq2SeqLM.from_pretrained(path) for path, lang in zip(args.teacher_path, args.teacher_lang)})

loss_weights = {'ce': args.loss_weights[0], 'kl': args.loss_weights[1], 'cosine': args.loss_weights[2]}

distiller = DistillerBilingTeachers(
    teachers=teachers,
    student_size=args.student_size,
    loss_weights=loss_weights,
    lr=args.lr,
    weight_decay=args.weight_decay,
    random_initialized_student=args.random_initialized_student,
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

trainer.test()
