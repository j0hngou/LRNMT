import sys
import pytorch_lightning as pl
import argparse

sys.path.append('../')

from distiller_biling_teachers import DistillerBilingTeachers, DistillerEnItTeachers
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
parser.add_argument('--dataset_names', nargs='+', type=str,
                    help='The path(or huggingface path) to the datasets. If there are multiple, they should be separated by a space.',
                    default=["j0hngou/ccmatrix_en-it"])
parser.add_argument('--loss_weights', nargs='+', type=float, default=[1 / 2, 1 / 2, 0], help='The weights to use for the loss. \
                    loss_weights format: [CE, KL, Cosine]')
parser.add_argument('--lr', type=float, default=2e-5, help='The learning rate.')
parser.add_argument('--weight_decay', type=float, default=0.0, help='The weight decay.')
parser.add_argument('--batch_size', type=int, default=5, help='The batch size.')
parser.add_argument('--max_epochs', type=int, default=10, help='The maximum number of epochs.')
parser.add_argument('--fp16', action='store_true', default=False, help='Whether to use mixed precision training.')
parser.add_argument('--wandb_project', type=str, default='distiller', help='The wandb project name.')
parser.add_argument('--val_check_interval', type=float, default=0.05, help='The validation check interval.')
parser.add_argument('--seed', type=int, default=123, help='The seed to use.')
parser.add_argument('--random_initialized_student', action='store_true', help='Whether the student is random initialized. If not, the en-ro-t5-small model will be used.',
default=False)
parser.add_argument('--experiment_name', type=str, default='', help='The name of the experiment.')
parser.add_argument('--disable_dropout', action='store_true', help='Disables dropout in the student model.', default=True)
parser.add_argument('--schedule', type=str, default='', help='The schedule to use for the KL loss decay', choices=['linear', 'cosine'])
parser.add_argument('--warmup_steps', type=int, default=1000, help='The number of warmup steps for the KL loss decay.')
parser.add_argument('--decay_epochs', type=int, default=5, help='The number of epochs to decay the KL loss.')

args = parser.parse_args()

pl.seed_everything(args.seed)

if len(args.dataset_names) == 1:
    dm = MTDistillationDatamodule(batch_size=args.batch_size,
                                  dataset_names=args.dataset_names,
                                  source_target_pair=[("en", "it")],
                                  )
    args.teacher_path = ['din0s/t5-base_fr-finetuned-en-to-it',
                         'din0s/t5-base_ro-finetuned-en-to-it']
    args.teacher_lang = ['en-fr', 'en-ro']
    init_path = 'din0s/t5-small-ro-finetuned-en-to-it'
else:
    dm = MTDistillationDatamodule(batch_size=args.batch_size,
                                  )
    init_path = "din0s/t5-small-finetuned-en-to-it-b32"

dm.setup()

num_train_batches = len(dm.train_dataloader())

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

if len(args.dataset_names) == 1:
    distiller = DistillerEnItTeachers(
        teachers=teachers,
        loss_weights=loss_weights,
        lr=args.lr,
        weight_decay=args.weight_decay,
        random_initialized_student=args.random_initialized_student,
        disable_dropout=args.disable_dropout,
        precision=16 if args.fp16 else 32,
        schedule=args.schedule,
        warmup_steps=args.warmup_steps,
        decay_epochs=args.decay_epochs,
        init_path=init_path
    )
else:
    distiller = DistillerBilingTeachers(
        teachers=teachers,
        loss_weights=loss_weights,
        lr=args.lr,
        weight_decay=args.weight_decay,
        random_initialized_student=args.random_initialized_student,
        disable_dropout=args.disable_dropout,
        precision=16 if args.fp16 else 32,
        schedule=args.schedule,
        warmup_steps=args.warmup_steps,
        decay_epochs=args.decay_epochs,
        init_path=init_path
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

trainer.test(distiller, dm)
