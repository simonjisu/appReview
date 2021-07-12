import spacy
from spacy import displacy

import re
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter, defaultdict

import pickle
from tqdm.notebook import tqdm
from pathlib import Path
from wordcloud import WordCloud

from gensim.models import TfidfModel, Word2Vec
from gensim.corpora import Dictionary
from gensim.models import LdaModel, CoherenceModel

from typing import List, Tuple, Union

def draw_rate_plot(df, title, normalize=False):
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    rate_size = df.pivot_table(index="rate", aggfunc="size")
    if normalize:
        rate_size = (rate_size / rate_size.sum()).reset_index().rename(columns={0: "ratio"})
        sns.barplot(x="rate", y="ratio", data=rate_size, ax=ax)
        rate_size = rate_size.set_index("rate")["ratio"]
    else:
        sns.countplot(x="rate", data=df, ax=ax)
    ax.set_title(title)
    for i, rate in enumerate(rate_size, 1):
        s = f"{rate}" if isinstance(rate, int) else f"{rate:.4f}"
        ax.text(i-1-0.1, rate+rate*0.01, s=s)
    plt.show()
    
get_info = lambda x: (x.lower_, x.lemma_.lower(), x.pos_, x.tag_, x.is_stop)

def process_num_tkns(x: tuple):
    x_lower, x_lemma, x_pos, x_tag = x
    if (x_pos == "NUM") or (x_tag == "CD"):
        return ("[num]", "[num]", x_pos, x_tag)
    else:
        return (x_lower, x_lemma, x_pos, x_tag)
    
def process_remove_s(x: str):
    return x.replace('"', "").replace("-", "")

def preprocessing(txt: spacy.tokens.doc.Doc, stopwords: Union[bool, list]):    
    # tokenize & remove punctuations
    if isinstance(stopwords, bool):
        tkns = [get_info(tkn)[:-1] for tkn in txt if not ((tkn.is_stop) or (tkn.pos_ in ["PUNCT"]) or (tkn.tag_ in [",", "."]))]
    else:
        # specify the stopwords
        tkns = [get_info(tkn)[:-1] for tkn in txt if not ((tkn.tag_ in stopwords) or (tkn.pos_ in stopwords))]
    
    # change NUM to special token [num]
    tkns = [process_num_tkns(x) for x in tkns]
    return tkns

def preprocessing_pipeline(data_path, spacy_nlp, df, filename, save=False):
    if save:
        with (data_path / filename).open("wb") as file:
            pickle.dump(list(df["text"].apply(process_remove_s).apply(spacy_nlp)), file)
        return None
    else:
        with (data_path / filename).open("rb") as file:
            doc_data = pickle.load(file)
        return doc_data
    
def get_tokens(doc_data):
    tkns = []
    stopwords = [",", ".", "PUNCT"]
    for text in tqdm(doc_data, total=len(doc_data), desc=f"stopwords: {stopwords}"):
        tkns.append(preprocessing(text, stopwords=stopwords))

    tkns_remove_stop = []
    for text in tqdm(doc_data, total=len(doc_data), desc="stopwords: spacy reference"):
        tkns_remove_stop.append(preprocessing(text, stopwords=True))
    return tkns, tkns_remove_stop

def join_tkn_func(x: tuple, idx1: int, idx2: int):
    """
    idx1: token(0) or lemma token(1)
    idx2: pos(2) or tag(3)
    
    - pos: coarse-grained tags, https://universaldependencies.org/docs/u/pos/
    - tag: fine-grained part-of-speech tags
    """
    return f"{x[idx1]}__{x[idx2]}"

def save2documents(tkns: List[Tuple[str]], sv_path: Union[str, Path], idx1: int, idx2: int):
    """
    tkns: list of tuple tokens, (token, lemma, pos, tag, is_stop)
    filename: save filename
    idx1: token(0) or lemma token(1)
    idx2: pos(2) or tag(3)
    
    - pos: coarse-grained tags, https://universaldependencies.org/docs/u/pos/
    - tag: fine-grained part-of-speech tags
    """
    sv_path = Path(sv_path)
    filename = sv_path.name
    with sv_path.open("w", encoding="utf-8") as file:
        for doc in tqdm(tkns, total=len(tkns), desc=f"processing: {filename}"):
            print(" ".join([join_tkn_func(tkn, idx1, idx2) for tkn in doc]), file=file)
             
def to_token(txt, spacy_nlp, token=True, pos=True):
    txt = spacy_nlp(txt)
    idx1 = 0 if token else 1
    idx2 = 2 if pos else 3
    tkns = [get_info(t)[:-1] for t in txt]
    tkns = [process_num_tkns(t) for t in tkns]
    return [join_tkn_func(t, idx1, idx2) for t in tkns]

def save2document_all(tkns, tkns_remove_stop, poststr=None):
    poststr = "" if poststr is None else f"_{poststr}"
    save2documents(tkns, f"./data/token_pos{poststr}.txt", 0, 2)
    save2documents(tkns, f"./data/token_tag{poststr}.txt", 0, 3)
    save2documents(tkns, f"./data/lemma_pos{poststr}.txt", 1, 2)
    save2documents(tkns, f"./data/lemma_tag{poststr}.txt", 1, 3)
    save2documents(tkns_remove_stop, f"./data/token_pos_rm_stop{poststr}.txt", 0, 2)
    save2documents(tkns_remove_stop, f"./data/token_tag_rm_stop{poststr}.txt", 0, 3)
    save2documents(tkns_remove_stop, f"./data/lemma_pos_rm_stop{poststr}.txt", 1, 2)
    save2documents(tkns_remove_stop, f"./data/lemma_tag_rm_stop{poststr}.txt", 1, 3)

tokenizer_token_pos = lambda x: to_token(x, spacy_nlp, token=True, pos=True)
tokenizer_token_tag = lambda x: to_token(x, spacy_nlp, token=True, pos=False)
tokenizer_lemma_pos = lambda x: to_token(x, spacy_nlp, token=False, pos=True)
tokenizer_lemma_tag = lambda x: to_token(x, spacy_nlp, token=False, pos=False)


def load_data(path: Union[str, Path]):
    path = Path(path)
    with path.open("r", encoding="utf-8") as file:
        data = [line.strip().split() for line in file.readlines()]
    return data

def get_pos_tokens(counter, tags, length=15):
    res = {}
    for w, c in counter.most_common():
        word, tag = w.split("__")
        if tag in tags:
            res[word] = c
        if (length > 0) and (len(res) >= length):
            break
    return dict(sorted(res.items(), key=lambda x: x[1], reverse=True))

def draw_wordcloud(unique_rates, n_sample_rates, clouds, typ):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for r, n, ax in zip(unique_rates, n_sample_rates, axes.flatten()):
        cloud = clouds[r][typ]
        ax.imshow(cloud.to_array(), interpolation="bilinear")
        ax.set_title(f"{typ.upper()} Word Cloud at rating {r}(n_sample={n})", fontsize=14)
        ax.axis(False)
    axes[-1][-1].axis(False)
    plt.tight_layout()
    plt.show()
    
def get_clouds(font_path, unique_rates, n_sample_rates, rates, data):
    word_counts = defaultdict(dict)
    for r in unique_rates:
        word_counts[r]["counter"] = Counter()
    for doc, r in zip(data, rates):
        word_counts[r]["counter"].update(doc)

    length = -1
    for r in unique_rates:
        word_counts[r]["adj"] = get_pos_tokens(word_counts[r]["counter"], tags=["ADJ"], length=length)
        word_counts[r]["noun"] = get_pos_tokens(word_counts[r]["counter"], tags=["NOUN"], length=length)

    # remove common words in 5 ratings
    adj_arrays = []
    noun_arrays = []
    for r in unique_rates:
        adj_arrays.append(np.array(list(word_counts[r]["adj"].keys())))
        noun_arrays.append(np.array(list(word_counts[r]["noun"].keys())))

    for r in unique_rates:
        x = list(range(5))
        x.remove(r-1)
        word_counts[r]["noncommon_adj"] = adj_arrays[r-1][np.concatenate(
            [np.isin(adj_arrays[r-1], adj_arrays[i]).reshape(1, -1) for i in x], axis=0).sum(0) != 4]
        word_counts[r]["noncommon_noun"] = noun_arrays[r-1][np.concatenate(
            [np.isin(noun_arrays[r-1], noun_arrays[i]).reshape(1, -1) for i in x], axis=0).sum(0) != 4]

    clouds = defaultdict(dict)
    length = 100
    for r in unique_rates:
        adj_cnt = Counter({w: word_counts[r]["adj"][w] for w in word_counts[r]["noncommon_adj"]}).most_common(length)
        noun_cnt = Counter({w: word_counts[r]["noun"][w] for w in word_counts[r]["noncommon_noun"]}).most_common(length)
        clouds[r]["adj"] = WordCloud(
            font_path=font_path, width=800, height=800, background_color="white", scale=0.9
        ).generate_from_frequencies(dict(adj_cnt))
        clouds[r]["noun"] = WordCloud(
            font_path=font_path, width=800, height=800, background_color="white", scale=0.9
        ).generate_from_frequencies(dict(noun_cnt))
    
    return {"word_count": word_counts, "adj_arrays": adj_arrays, "noun_arrays": noun_arrays, "clouds": clouds}