import pandas as pd

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

def get_word_spans(text):
    word_start = True

    starts, ends = [], []
    end = -1

    for i in range(len(text)):

        if text[i].isspace():
            if end != -1:
                ends.append(end)
                end = -1

            word_start=True
            continue

        if word_start==True:
            starts.append(i)
            word_start=False

        end = i + 1

    if len(starts) > len(ends):
        ends.append(end)

    return list(zip(starts, ends))

def make_predictionstring(row, word_spans):
    start = row['discourse_start']
    end = row['discourse_end']

    predictionstring = []

    for idx, (span_start, span_end) in enumerate(word_spans):
        if min(span_end, end) - max(span_start, start) > 0:
            predictionstring.append(str(idx))
    
    return ' '.join(predictionstring)

def prepare_data():

    df = pd.read_csv('data/persuade_corpus.csv', low_memory=False)
    df['discourse_text'] = df['discourse_text'].transform(clean_text)
    
    text_df = df[['essay_id', 'full_text']].drop_duplicates('essay_id').reset_index(drop=True)
    text_df['full_text'] = text_df['full_text'].transform(clean_text)  

    df = df.drop('full_text', axis=1)

    LABEL2EFFEC = ('Adequate', 'Effective', 'Ineffective')

    df = df[df['discourse_type'] != "Unannotated"]
    df = df[df['discourse_effectiveness'].isin(LABEL2EFFEC)]

    # make prediction strings like in fb1 comp
    text_dict = text_df.set_index('essay_id')['full_text'].to_dict()
    spans_dict = {k: get_word_spans(v) for k,v in text_dict.items()}
    
    df['predictionstring'] = df.apply(lambda row: make_predictionstring(row, spans_dict[row['essay_id']]), axis=1)

    train_df = df[df['competition_set'] == 'train'].reset_index(drop=True)
    valid_df = df[df['test_split_feedback_1'] == 'Public'].reset_index(drop=True)
    test_df = df[df['test_split_feedback_1'] == 'Private'].reset_index(drop=True)

    text_df.to_csv('data/text_df.csv', index=False)
    train_df.to_csv('data/train_df.csv', index=False)
    valid_df.to_csv('data/valid_df.csv', index=False)
    test_df.to_csv('data/test_df.csv', index=False)

def get_data():

    text_df = pd.read_csv('data/text_df.csv', low_memory=False)
    text_df = text_df[~text_df['essay_id'].isin({'AAAOPP13416000014141', '5532021152126'})]
    text_df.set_index('essay_id', inplace=True)

    train_df = pd.read_csv('data/train_df.csv', low_memory=False)
    valid_df = pd.read_csv('data/valid_df.csv', low_memory=False)
    test_df = pd.read_csv('data/test_df.csv', low_memory=False)

    text_df['len'] = text_df.apply(lambda row: len(row['full_text']), axis=1)
    train_df = train_df[train_df['essay_id'].isin(text_df.index)].reset_index(drop=True)

    return text_df, train_df, valid_df, test_df