import torchtext.datasets as datasets
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchtext.vocab import build_vocab_from_iterator

from utils import *


def get_src_iter(data_iter):
    for data_sample in data_iter:
        yield data_sample[0]

def get_tgt_iter(data_iter):
    for data_sample in data_iter:
        yield data_sample[1]

def get_vocab(train_iter, tokenizer, conf):
    special_tokens = conf["special_tokens"]
    unk_idx = special_tokens[conf["unk_token"]]
    vocab = build_vocab_from_iterator(
        map(tokenizer, train_iter),
        min_freq = conf["min_freq"],
        specials = list(special_tokens.keys())
    )
    vocab.set_default_index(unk_idx)
    return vocab


def load_multi30k(src_lang, tgt_lang):
    train_iter, valid_iter, test_iter =  \
        datasets.Multi30k(language_pair=(src_lang, tgt_lang))
    return list(train_iter), list(valid_iter), list(test_iter)

def load_iwslt2016(src_lang, tgt_lang):
    train_iter, valid_iter, test_iter =  \
        datasets.IWSLT2016(language_pair=(src_lang, tgt_lang))
    return list(train_iter), list(valid_iter), list(test_iter)

def load_iwslt2017(src_lang, tgt_lang):
    train_iter, valid_iter, test_iter =  \
        datasets.IWSLT2016(language_pair=(src_lang, tgt_lang))
    return list(train_iter), list(valid_iter), list(test_iter)


def get_dataloader(data, src_transform, tgt_transform, conf):
    def collate_fn(batch):
        PAD_IDX = conf["special_tokens"][conf["pad_token"]]
        src_batch, tgt_batch = [], []
        for src_sample, tgt_sample in batch:
            src_batch.append(src_transform(src_sample.rstrip("\n")))
            tgt_batch.append(tgt_transform(tgt_sample.rstrip("\n")))
        src_batch = pad_sequence(src_batch, padding_value=PAD_IDX, batch_first=True)
        tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX, batch_first=True)
        return src_batch, tgt_batch
    return DataLoader(data, batch_size=conf["batch_size"], collate_fn=collate_fn)