import torch

class Encoder(torch.nn.Module):
    def __init__(self,
                 input_dim,
                 emb_dim,
                 hid_dim,
                 n_layers,
                 kernel_size,
                 dropout,
                 device,
                 max_length = 100):
        super().__init__()
        
        assert kernel_size % 2 == 1, "Kernel size must be odd!"
        
        self.device = device
        
        self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(device)
        
        self.tok_embedding = torch.nn.Embedding(input_dim, emb_dim)
        self.pos_embedding = torch.nn.Embedding(input_dim, emb_dim)
        
        self.emb2hid = torch.nn.Linear(emb_dim, hid_dim)
        self.hid2emb = torch.nn.Linear(hid_dim, emb_dim)
        
        self.convs = torch.nn.ModuleList([
            torch.nn.Conv1d(in_channels = hid_dim, 
                            out_channels = 2 * hid_dim, 
                            kernel_size = kernel_size, 
                            padding = (kernel_size - 1) // 2)
        for _ in range(n_layers)])
        self.dropout = torch.nn.Dropout(dropout)
        
    def forward(self, src):
        batch_size = src.shape[0]
        src_len = src.shape[1]
        
        #create position tensor
        pos = torch.arange(0, src_len).repeat(batch_size, 1).to(self.device)
        
        #embed tokens and positions
        tok_embedded = self.tok_embedding(src)
        pos_embedded = self.pos_embedding(pos)

        #combine embeddings by elementwise summing
        embedded = self.dropout(tok_embedded + pos_embedded)
        
        #pass embedded through linear layer to convert from emb dim to hid dim
        conv_input = self.emb2hid(embedded)
        
        #permute for convolutional layer
        conv_input = conv_input.permute(0, 2, 1) 
        
        #begin convolutional blocks...
        for conv in self.convs:
            conved = conv(self.dropout(conv_input))
            conved = torch.nn.functional.glu(conved, dim = 1)
            #apply residual connection
            conved = (conved + conv_input) * self.scale
            #set conv_input to conved for next loop iteration
            conv_input = conved
        
        #permute and convert back to emb dim
        conved = self.hid2emb(conved.permute(0, 2, 1))
        #elementwise sum output (conved) and input (embedded) to be used for attention
        combined = (conved + embedded) * self.scale
        return conved, combined


class Decoder(torch.nn.Module):
    def __init__(self, 
                 output_dim, 
                 emb_dim, 
                 hid_dim, 
                 n_layers, 
                 kernel_size, 
                 dropout, 
                 trg_pad_idx, 
                 device,
                 max_length = 100):
        super().__init__()
        
        self.kernel_size = kernel_size
        self.trg_pad_idx = trg_pad_idx
        self.device = device
        
        self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(device)
        
        self.tok_embedding = torch.nn.Embedding(output_dim, emb_dim)
        self.pos_embedding = torch.nn.Embedding(output_dim, emb_dim)
        
        self.emb2hid = torch.nn.Linear(emb_dim, hid_dim)
        self.hid2emb = torch.nn.Linear(hid_dim, emb_dim)
        
        self.attn_hid2emb = torch.nn.Linear(hid_dim, emb_dim)
        self.attn_emb2hid = torch.nn.Linear(emb_dim, hid_dim)
        
        self.fc_out = torch.nn.Linear(emb_dim, output_dim)
        
        self.convs = torch.nn.ModuleList([
            torch.nn.Conv1d(in_channels = hid_dim, 
                            out_channels = 2 * hid_dim, 
                            kernel_size = kernel_size)
        for _ in range(n_layers)])
        
        self.dropout = torch.nn.Dropout(dropout)
      

    def calculate_attention(self, embedded, conved, encoder_conved, encoder_combined):      
        #permute and convert back to emb dim
        conved_emb = self.attn_hid2emb(conved.permute(0, 2, 1))
        combined = (conved_emb + embedded) * self.scale
        energy = torch.matmul(combined, encoder_conved.permute(0, 2, 1))
        attention = torch.nn.functional.softmax(energy, dim=2)
        attended_encoding = torch.matmul(attention, encoder_combined)
        #convert from emb dim -> hid dim
        attended_encoding = self.attn_emb2hid(attended_encoding)
        #apply residual connection
        attended_combined = (conved + attended_encoding.permute(0, 2, 1)) * self.scale
        return attention, attended_combined
        
    def forward(self, trg, encoder_conved, encoder_combined):
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        #create position tensor
        pos = torch.arange(0, trg_len).repeat(batch_size, 1).to(self.device)
        #embed tokens and positions
        tok_embedded = self.tok_embedding(trg)
        pos_embedded = self.pos_embedding(pos)
        #combine embeddings by elementwise summing
        embedded = self.dropout(tok_embedded + pos_embedded)
        #pass embedded through linear layer to go through emb dim -> hid dim
        conv_input = self.emb2hid(embedded)
        #permute for convolutional layer
        conv_input = conv_input.permute(0, 2, 1) 
        batch_size = conv_input.shape[0]
        hid_dim = conv_input.shape[1]
        for i, conv in enumerate(self.convs):
            #apply dropout
            conv_input = self.dropout(conv_input)
            #need to pad so decoder can't "cheat"
            padding = torch.zeros(batch_size, 
                                  hid_dim, 
                                  self.kernel_size - 1).fill_(self.trg_pad_idx).to(self.device)
            padded_conv_input = torch.cat((padding, conv_input), dim = 2)
            #pass through convolutional layer
            conved = conv(padded_conv_input)
            #pass through GLU activation function
            conved = torch.nn.functional.glu(conved, dim = 1)
            #calculate attention
            attention, conved = self.calculate_attention(embedded, 
                                                         conved, 
                                                         encoder_conved, 
                                                         encoder_combined)
            #apply residual connection
            conved = (conved + conv_input) * self.scale
            #set conv_input to conved for next loop iteration
            conv_input = conved
            
        conved = self.hid2emb(conved.permute(0, 2, 1))
        output = self.fc_out(self.dropout(conved))
        return output, attention


class Seq2Seq(torch.nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        
    def forward(self, src, trg):
        encoder_conved, encoder_combined = self.encoder(src)
        output, attention = self.decoder(trg, encoder_conved, encoder_combined)
        return output, attention