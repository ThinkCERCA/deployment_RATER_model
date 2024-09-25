import torch
import re
import numpy as np
import random
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader

LABEL2TYPE = ('Lead', 'Position', 'Claim', 'Counterclaim', 'Rebuttal',
              'Evidence', 'Concluding Statement')

TYPE2LABEL = {t: l for l, t in enumerate(LABEL2TYPE)}

LABEL2EFFEC = ('Adequate', 'Effective', 'Ineffective')
EFFEC2LABEL = {t: l for l, t in enumerate(LABEL2EFFEC)}

import codecs
from text_unidecode import unidecode

def replace_encoding_with_utf8(error):
    return error.object[error.start : error.end].encode("utf-8"), error.end


def replace_decoding_with_cp1252(error):
    return error.object[error.start : error.end].decode("cp1252"), error.end


# Register the encoding and decoding error handlers for `utf-8` and `cp1252`.
codecs.register_error("replace_encoding_with_utf8", replace_encoding_with_utf8)
codecs.register_error("replace_decoding_with_cp1252", replace_decoding_with_cp1252)

def resolve_encodings_and_normalize(text):
    """Resolve the encoding problems and normalize the abnormal characters."""
    text = (
        text.encode("raw_unicode_escape")
        .decode("utf-8", errors="replace_decoding_with_cp1252")
        .encode("cp1252", errors="replace_encoding_with_utf8")
        .decode("utf-8", errors="replace_decoding_with_cp1252")
    )
    text = unidecode(text)
    return text

def clean_text(text):
    text = text.replace(u'\xa0', u' ')
    text = text.replace(u'\x85', u'\n')
    text = text.strip()
    text = resolve_encodings_and_normalize(text)

    return text

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, cfg):

        self.texts = texts
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):

        text = clean_text(self.texts[index])

        tokens = self.tokenizer(
            text,
            add_special_tokens=True,
            return_offsets_mapping=True
        )

        input_ids = torch.LongTensor(tokens['input_ids'])
        attention_mask = torch.LongTensor(tokens['attention_mask'])
        offset_mapping = np.array(tokens['offset_mapping'])
        offset_mapping = self.strip_offset_mapping(text, offset_mapping)

        # token slices of words
        woff = self.get_word_offsets(text)
        toff = offset_mapping
        wx1, wx2 = woff.T
        tx1, tx2 = toff.T
        ix1 = np.maximum(wx1[..., None], tx1[None, ...])
        ix2 = np.minimum(wx2[..., None], tx2[None, ...])
        ux1 = np.minimum(wx1[..., None], tx1[None, ...])
        ux2 = np.maximum(wx2[..., None], tx2[None, ...])
        ious = (ix2 - ix1).clip(min=0) / (ux2 - ux1 + 1e-12)
        assert (ious > 0).any(-1).all()

        word_boxes = []
        for row in ious:
            inds = row.nonzero()[0]
            word_boxes.append([inds[0], 0, inds[-1] + 1, 1])
        word_boxes = torch.FloatTensor(word_boxes)

        return dict(text=text, input_ids=input_ids, attention_mask=attention_mask, word_boxes=word_boxes)

    def strip_offset_mapping(self, text, offset_mapping):
        ret = []
        for start, end in offset_mapping:
            match = list(re.finditer('\\S+', text[start:end]))
            if len(match) == 0:
                ret.append((start, end))
            else:
                span_start, span_end = match[0].span()
                ret.append((start + span_start, start + span_end))
        return np.array(ret)

    def get_word_offsets(self, text):
        matches = re.finditer("\\S+", text)
        spans = []
        words = []
        for match in matches:
            span = match.span()
            word = match.group()
            spans.append(span)
            words.append(word)
        assert tuple(words) == tuple(text.split())
        return np.array(spans)


class CustomCollator(object):
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, samples):
        batch_size = len(samples)
        assert batch_size == 1, f'Only batch_size=1 supported, got batch_size={batch_size}.'

        sample = samples[0]

        max_seq_length = len(sample['input_ids'])
        padded_length = max_seq_length

        input_shape = (1, padded_length)
        input_ids = torch.full(input_shape,
                               self.pad_token_id,
                               dtype=torch.long)
        attention_mask = torch.zeros(input_shape, dtype=torch.long)

        seq_length = len(sample['input_ids'])
        input_ids[0, :seq_length] = sample['input_ids']
        attention_mask[0, :seq_length] = sample['attention_mask']

        text = sample['text']
        word_boxes = sample['word_boxes']

        return dict(text=text,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    word_boxes=word_boxes)

class TextDataModule(pl.LightningDataModule):
    def __init__(
        self,
        texts = None,
        tokenizer = None,
        cfg = None,
    ):
        super().__init__()
        self.texts = texts
        self.tokenizer = tokenizer
        self.cfg = cfg

    def setup(self, stage):
        if stage == 'predict':
            self.predict_dataset = TextDataset(self.texts, self.tokenizer, self.cfg)
        else:
            raise Exception()

    def predict_dataloader(self):
        custom_collator = CustomCollator(self.tokenizer.pad_token_id)
        return DataLoader(self.predict_dataset, **self.cfg["val_loader"], collate_fn=custom_collator)
