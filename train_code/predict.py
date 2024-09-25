from code.data import get_data
from code.dataset import TextDataModule
from code.model import TextModel

import argparse
import os
import yaml
import shutil
import pickle

import torch
import pytorch_lightning as pl

from transformers import AutoTokenizer

torch.set_float32_matmul_precision('medium')
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

os.makedirs('preds', exist_ok=True)

def parse_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--ckpt', default=None, action='store', required=True
    )

    parser.add_argument(
        '--test', default=False, action='store_true'
    )

    parser.add_argument(
        '--xlarge', default=False, action='store_true'
    )

    return parser.parse_known_args()[0]

opt = parse_opt()

if opt.xlarge:
    with open('runs/swa/deberta_xlarge.yml', 'r') as f:
        cfg = yaml.safe_load(f)
else:
    with open('runs/swa/deberta_large.yml', 'r') as f:
        cfg = yaml.safe_load(f)

cfg['skip_validation'] = 0

text_df, train_df, valid_df, test_df = get_data()
cfg['dataset_size'] = len(train_df)

model_name = cfg['model']['model_name']

tokenizer = AutoTokenizer.from_pretrained(model_name)

if opt.test:
    datamodule = TextDataModule(text_df = text_df, train_df=train_df, valid_df=test_df, predict_df=test_df, tokenizer=tokenizer, cfg=cfg)
else:
    datamodule = TextDataModule(text_df = text_df, train_df=train_df, valid_df=valid_df, predict_df=valid_df, tokenizer=tokenizer, cfg=cfg)
model = TextModel(cfg)

if opt.ckpt:
    state_dict = torch.load(opt.ckpt)['state_dict']

    for key in list(state_dict.keys()):
        state_dict[key] = state_dict.pop(key)

    model.load_state_dict(state_dict, strict=True)

trainer = pl.Trainer(
    logger=False,
    accelerator="gpu",
    max_epochs=cfg['epochs'],
    callbacks=[],
    **cfg['trainer'],
)


preds = trainer.predict(model, datamodule=datamodule)

filename = opt.ckpt.replace('runs/swa/', 'preds/').replace('.ckpt','.p')

if opt.test:
    filename = filename.replace('.p', '_test.p')

pickle.dump(preds, open(filename, 'wb'))