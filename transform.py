import torch



class TextTransform(torch.nn.Module):
    def __init__(self, tokenizer, vocab, conf):
        super(TextTransform, self).__init__()
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.SOS_IDX = conf["special_tokens"][conf["sos_token"]]
        self.EOS_IDX = conf["special_tokens"][conf["eos_token"]]
    
    def forward(self, sentence):
        tokens = self.tokenizer(sentence)
        token_ids = self.vocab(tokens)
        return torch.cat((torch.tensor([self.SOS_IDX]),
                      torch.tensor(token_ids),
                      torch.tensor([self.EOS_IDX])))

