from train_code.code.dataset import TextDataModule
from train_code.code.pres import make_preds
from train_code.code.model import TextModel
from train_code.word2sent import convert_labels_to_sentence_level
from train_code.filter import filter_claim_and_evidence
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import yaml
import pandas as pd

import torch
import pytorch_lightning as pl

from transformers import AutoTokenizer

torch.set_float32_matmul_precision('medium')
os.environ['TOKENIZERS_PARALLELISM'] = 'true'



with open('deberta_large_025.yml', 'r') as f:
    cfg = yaml.safe_load(f)

cfg['skip_validation'] = 0

tokenizer = AutoTokenizer.from_pretrained('tokenizer/tokenizer')

trainer = pl.Trainer(logger=False, accelerator="cpu",**cfg['trainer'])

model = TextModel(cfg)

state_dict = torch.load('best_deberta_large.ckpt')['state_dict']

for key in list(state_dict.keys()):
    state_dict[key] = state_dict.pop(key)

model.load_state_dict(state_dict, strict=True)


def predict_single(model, text):
    datamodule = TextDataModule(texts=[text], tokenizer=tokenizer, cfg=cfg)
    preds = trainer.predict(model, datamodule=datamodule)
    pred_df = make_preds(preds, ['NO_ID'], 0.33)

    return pred_df


LABEL2TYPE = ('Lead', 'Position', 'point', 'Counterclaim', 'Rebuttal',
              'Evidence', 'Concluding Statement')

TYPE2LABEL = {t: l for l, t in enumerate(LABEL2TYPE)}


# Assuming your DataFrame is called 'df'
# If not, replace 'df' with the name of your DataFrame

def df_to_json(df):
    json_list = []
    for _, row in df.iterrows():
        # Get the label type from the numeric label
        label_type = LABEL2TYPE[row['discourse_type']] if isinstance(row['discourse_type'], int) else row[
            'discourse_type']

        # Convert 'Position' to 'Claim' if needed
        if label_type == 'Position':
            label_type = 'Claim'

        json_obj = {
            "label": label_type,
            "start": int(row['start']),
            "end": int(row['end'])
        }
        json_list.append(json_obj)
    return json_list

app = Flask(__name__)
CORS(app)
@app.route("/", methods=['GET', 'POST'])

def run():
    if request.method == 'POST':
        data_fz = request.get_json()
        if data_fz != None:
            content = data_fz['content']
        else:
            return jsonify({'type': 'Error', 'result': ''})  ####没有接收到数据
    else:
        return jsonify({'type': 'Error', 'result': ''})
    label = convert_labels_to_sentence_level(content, df_to_json(predict_single(model, content)))
    label = filter_claim_and_evidence(label)


    return jsonify({'type': 'success', 'result':label})
if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080, debug=False)