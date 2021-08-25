class Tokenizer:
    def tokenize(self, sentence):
        raise NotImplementedError

class WhiteSpaceTokenizer(Tokenizer):
    def tokenizer(self, sentence):
        return sentence.split()

class SpacyTokenizer(Tokenizer):
    def __init__(self, lang):
        import spacy
        self.nlp = spacy.load(lang)
    
    def tokenize(self, sentence):
        return [tok.text for tok in self.nlp.tokenizer(sentence)]

def load_tokenizer(name, lang):
    if name == 'whitespace':
        return WhiteSpaceTokenizer()
    elif name == "moses":
        from sacremoses import MosesTokenize
        return MosesTokenize()
    elif name == "spacy":
        return SpacyTokenizer(lang)
    else:
        raise ValueError("Unknown tokenizer: {}".format(name))