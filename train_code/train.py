from code.data import get_data
from code.dataset import TextDataModule
from code.model import TextModel

import argparse
import os
import yaml
import shutil

import torch
import pytorch_lightning as pl

from transformers import AutoTokenizer

torch.set_float32_matmul_precision('medium')
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--name', default=None, action='store', required=True
    )

    parser.add_argument(
        '--yaml', default=None, action='store', required=True
    )

    parser.add_argument(
        '--ckpt', default=None, action='store', required=False
    )

    parser.add_argument(
        '--no_wandb', default=False, action='store_true'
    )

    parser.add_argument(
        '--project', default='tlab', action='store', required=False
    )

    return parser.parse_known_args()[0]

opt = parse_opt()

with open(opt.yaml, 'r') as f:
    cfg = yaml.safe_load(f)

text_df, train_df, valid_df, test_df = get_data()
cfg['dataset_size'] = len(train_df)

root_dir = 'runs/' + opt.name

os.makedirs(root_dir, exist_ok=True)
shutil.copyfile(opt.yaml, root_dir + '/' + opt.yaml.split('/')[-1])
shutil.copyfile('train.py', root_dir + '/train.py')

model_name = cfg['model']['model_name']

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained(root_dir + '/tokenizer/')

datamodule = TextDataModule(text_df = text_df, train_df=train_df, valid_df=valid_df, tokenizer=tokenizer, cfg=cfg)
model = TextModel(cfg)
torch.save(model.config, root_dir + '/config.pth')

if opt.ckpt:
    state_dict = torch.load(opt.ckpt)['state_dict']

    for key in list(state_dict.keys()):
        state_dict[key] = state_dict.pop(key)

    model.load_state_dict(state_dict, strict=True)

loss_weights = pl.callbacks.ModelCheckpoint(
    dirpath=root_dir + '/weights',
    filename='ws-{epoch}-{score:.4f}',
    monitor='score',
    save_weights_only=True,
    save_top_k=100,
    mode='max',
    save_last=False,
)

loggers = False

if not opt.no_wandb:
    wandb_logger = pl.loggers.WandbLogger(project=opt.project, name=opt.name)
    try:
        wandb_logger.experiment.config.update(cfg)
    except:
        pass
    loggers = [wandb_logger]

trainer = pl.Trainer(
    logger=loggers,
    accelerator="gpu",
    max_epochs=cfg['epochs'],
    callbacks=[loss_weights],
    **cfg['trainer'],
)

if 'precision' in cfg['trainer'] and '16' in cfg['trainer']['precision']:
    model.scaler = trainer.precision_plugin.scaler
else:
    model.scaler = None

trainer.fit(model, datamodule=datamodule)