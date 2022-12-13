import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



class SelfAttention(nn.Module):

    def __init__(self, mask = None, droput = None):

        super(transformer_encoder_layer, self).__init__()
        
        self.mask = mask
        self.dropout = dropout



    def attention(self, query, key, value):

        dk = key.shape[-1] # used as a normalization constant 
        
        # quueries times keys.T
        score = torch.matmul(query, key.transpose(-1, -2)) #BxLxD
        scaled_score = score / torch.sqrt(dk)

        # Increase score to very large negative number for tokens that are masked.
        # such large negative number will have 0 exponential..
        if self.mask is not None:
            scaled_score.masked_fill(self.mask==0, -1e9)

        attention = F.softmax(scaled_score, dim=-1)

        # attention weights times the values
        Z = torch.matmul(attention, value)

        return Z, attention

    def forward(self, query, key, value, mask = None, dropout = None):

        return self.attention(query, key, value, mask, dropout)


class MultiheadAttention(nn.Module):

    def __init__(self, nhead, dmodel, dropout=0.1):
        super(MultiheadAttention, self).__init__()

        assert dmodel % nheads == 0
        self.dk = dmodel // nheads
        self.nheads = nheads

        self.Wq = nn.Linear(dmodel, dmodel)
        self.Wk = nn.Linear(dmodel, dmodel)
        self.Wv = nn.Linear(dmodel, dmodel)
        self.Wo = nn.Linear(dmodel, dmodel)

        self.selfatt = SelfAttention()
        
        self.dropout_value = dropout
        self.dropout = nn.Dropout(p = dropout)

    def forward(self, query, key, value, mask = None):

        if mask is not None: # apply same mask to all heads
            mask.unsqueeze(1)


        # keys, queries, values are the same sizes: (BxLxd_model)
        key, query, valu = self.Wk(key), self.Wq(query), self.Wv(value)

        # split up the operations into individual nheads (this is why we needed dmodel % nheads = 0)
        key = key.view(nbatches, -1, self.nheads, self.dk) # (B, L, nheads, dk)
        query = query.view(nbatches, -1, self.nheads, self.dk) # (B, L, nheads, dk)
        value = value.view(nbatches, -1, self.nheads, selfdk) # (B, L, nheads, dk)
        
        
        # swap dims
        key = key.permute(0,2,1,3) # (B, L, nheads, dk) --> (B, nheads, L dk)
        query = query.permute(0,2,1,3)
        value = value.permute(0,2,1,3)

        # calculate self attention adn enriched embedding z_i's
        z, self.attn = self.selfatt(
                query=query,
                key=key,
                value=value,
                mask=mask, 
                dropout=self.dropout
        ) # z: (B, nheads, L, dk) and attn: (B, nheads, L, L)
        
        # reshape tensors: (B, nheads, L, dk) --> z_concat: (B, L, nheads*dk)
        z_concat = z.permute(0, 2, 1, 3) # z: (B, nheads, L, dk) --> z_concat: (B, L, nheads, dk)
        z_concat = z_concat.contiuous() # z_concat: (B, L, nheads, dk) --> z_concat: (1, B*L*nheads*dk)
        z_concat = z_concat.view(nbatches, -1, self.nheads * self.dk) # z_concat: (1, B*L*nheads*dk) --> z_concat (B,L,nheads*dk)

        # Project z_concat with linear layer (Wo) to get final enriched embedding
        z_out = self.Wo(z_concat)
        return z_out



class FeedForwardNet(nn.Module):

    def __init__(self, dmodel, d_linear, dropout = 0.2):
        super(FeedForwardNet, self).__init__()
        self.W1 = nn.Linear(dmodel, d_linear)
        self.W2 = nn.Linear(d_linear, dmodel)
        self.dropout = nn.Dropout(dropout)

        self.relu = nn.ReLU()

    def forward(self, x):

        h = sel.relu(self.W1(x))
        output = self.dropout(self.W2(h))
        return output


class AddandNorm(nn.Module):

    def __init__(self, features, dropout = 0.2, epsilon = 1e-9):
        super(AddandNorm, self).__init__()
        self.layernorm = nn.LayerNorm(features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer_output):
        return self.layernorm(x + self.dropout(sublayer_output))

class TransformerEncoderLayer(nn.Module):


    def __init__(
            self,
            dmodel = 256,
            nhead = 4,
            dropout = 0.2,
            dlinear = 2048,
            mask = None
    ):

        super(TransformerEncoderLayer, self).__init__()

        self.MHA = MultiheadAttention(nheads, dmodel, dropout)
        self.add_norm1 = AddandNorm(dmodel, dropout)
        self.FFN = FeedForwardNet(dmodel, dlinear, dropout)
        self.add_norm2 = AddandNorm(dmodel, dropout)


    def forward(
            self,
            x,
            mask = None
    ):
        MHA_output = self.MHA(x, x, x, mask)
        norm_output1 = self.add_norm1(x, MHA_output)
        FFN_output = self.FFN(norm_output1)
        norm_output2 = self.add_norm2(norm_output1, FFN_output)
        return norm_output2


class TransformerEncoder(nn.Module):

    def __init__(
            self,
            num_layers: int = 3,
            emb_dim: int = 256,
            nhead: int = 4,
            dropout: float = 0.2,
            alphabet: int = 21
    ):

        super(TransformerEncoder, self).__init__()

        # encoder layer hps
        self.nhead = nhead
        self.emb_dim = emb_dim
        self.dropout = dropout

        # tranformer hps
        self.num_layers = num_layers
            
        # misc hps
        self.alphabet = alphabet
        
        # architecture
        self.encoder_layers = nn.ModuleList()

        for ii in range(self.num_layers):

            self.encoder_layers.append(
                    TransformerEncoderLayer(
                        dmodel = self.emb_dim,
                        nhead = self.nhead,
                        dropout = self.dropout,
                    )
            )
        
        
    def forward(self, x):

        for layer in self.encoder_layers:
            x = layer(x)

        return x
    


class context_transformer(nn.Module):


    def __init__(
            self,
            DEVICE: str = 'cuda',
            alphabet: int = 21,
            emb_dim: int = 256,
            time_dim: int = 256,
            in_feat_dim: int = 256,
            nhead: int = 4,
            num_layers: int = 3
    ):

        super(context_transformer, self).__init__()

        self.DEVICE = DEVICE

        # encoder layer hps
        self.in_feat_dim = in_feat_dim
        self.nhead = nhead

        # transformer hps
        self.num_layers = num_layers

        # misc hps
        self.alphabet = alphabet
        self.emb_dim = emb_dim
        self.time_dim = time_dim

        self.encoder_layer = nn.TransformerEncoderLayer(
                d_model = self.in_feat_dim,
                nhead = self.nhead
        )

        self.transformer_encoder = nn.TransformerEncoder(
                encoder_layer = self.encoder_layer,
                num_layers = self.num_layers
        )

        self.emb_tokens = nn.Embedding(self.alphabet, self.emb_dim)


    def pos_encoding(self, pos, channels):

        inv_freq = 1.0 / (
                10000
                ** (torch.arange(0, channels, 2) / channels)
        )

        pos_enc_a = torch.sin(pos.repeat(1, 1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(pos.repeat(1, 1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim = - 1)

        return pos_enc

    def compute_pos_encodings(self, x):

        batch_size = x.shape[0]
        seq_length = x.shape[1]

        pos_vector = torch.linspace(
                start = 1,
                end = seq_length,
                steps = seq_length,
                dtype = torch.long
        )

        # return pos vector for each sample
        pos = pos_vector.repeat(batch_size, 1)
        return pos.unsqueeze(-1)

    def forward(self, context_seq):

        # retrieve pos embeddings
        pos_emb = self.compute_pos_encodings(context_seq).to(self.DEVICE)

        # retrieve context sequence embedding
        seq_emb = self.emb_tokens(context_seq)

        # final input to transformer
        X_context = seq_emb + pos_emb

        # feed through transformer
        c = self.transformer_encoder(X_context)

        return c 

