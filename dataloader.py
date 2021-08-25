import torchtext.datasets as datasets
from torchtext.vocab import build_vocab_from_iterator



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

def get_src_iter(data_iter):
    for data_sample in data_iter:
        yield data_sample[0]

def get_tgt_iter(data_iter):
    for data_sample in data_iter:
        yield data_sample[1]

def get_vocab(train_iter, tokenizer, conf):
    special_symbols = conf["special_symbols"]
    unk_idx = special_symbols[conf["unk_token"]]
    vocab = build_vocab_from_iterator(
        map(tokenizer, train_iter),
        min_freq = conf["min_freq"],
        specials = list(special_symbols.keys())
    )
    vocab.set_default_index(unk_idx)
    return vocab

