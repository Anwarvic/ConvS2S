from utils import *
from model import *
from dataloader import *
from transform import TextTransform
from train import train



# load configurations
CONF = load_conf()
# add special characters to config
CONF["special_tokens"] = {
    CONF["unk_token"]: 0,
    CONF["pad_token"]: 1,
    CONF["sos_token"]: 2,
    CONF["eos_token"]: 3,
}

# set seed
set_seed(CONF['seed'])

# get device
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# load tokenizers
src_tokenizer = load_tokenizer(CONF['tokenizer'], CONF['src'])
tgt_tokenizer = load_tokenizer(CONF['tokenizer'], CONF['tgt'])

# load datset
train_data, valid_data, test_data = load_multi30k(CONF['src'], CONF['tgt'])

# load vocab
src_vocab = get_vocab( get_src_iter(train_data), src_tokenizer, CONF)
tgt_vocab = get_vocab( get_tgt_iter(train_data), tgt_tokenizer, CONF)

# data transforms
src_transform = TextTransform(src_tokenizer, src_vocab, CONF)
tgt_transform = TextTransform(tgt_tokenizer, tgt_vocab, CONF)

# get data loaders
train_dataloader = get_dataloader(train_data, src_transform, tgt_transform, CONF)
valid_dataloader = get_dataloader(valid_data, src_transform, tgt_transform, CONF)
test_dataloader = get_dataloader(test_data, src_transform, tgt_transform, CONF)


# create model
encoder = Encoder(input_dim = len(src_vocab),
                  emb_dim = CONF['embed_size'],
                  hid_dim = CONF['hidden_size'],
                  n_layers = CONF['encoder_layers'],
                  kernel_size = CONF['kernel_size'],
                  dropout = CONF['dropout'],
                  device = DEVICE,
                  max_length=CONF['max_length'],
)
decoder = Decoder(output_dim = len(tgt_vocab),
                  emb_dim = CONF['embed_size'],
                  hid_dim = CONF['hidden_size'],
                  n_layers = CONF['decoder_layers'],
                  kernel_size = CONF['kernel_size'],
                  dropout = CONF['dropout'],
                  tgt_pad_idx = CONF["special_tokens"][CONF['pad_token']],
                  device = DEVICE,
                  max_length=CONF['max_length'],
)
convs2s = Seq2Seq(encoder, decoder)

# create optimizer and criterion
optimizer = torch.optim.Adam(convs2s.parameters())
criterion = torch.nn.CrossEntropyLoss(ignore_index = CONF["special_tokens"][CONF['pad_token']])

# train
train(  convs2s,
        criterion,
        train_dataloader,
        valid_dataloader,
        optimizer,
        CONF["epochs"],
        CONF["clip"],
)