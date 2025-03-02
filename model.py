import torch
import torch.nn as nn
import math

class InputEmbedding(nn.Module): ## Converts a token into 512 dimensional vector
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model) ## Creates a lookuop table from tokens to  vectors

    def forward(self, x):
        return self.embedding(x)*math.sqrt(self.d_model)
    
class PositionEncoding(nn.Module): ## For giving the information about each token in the sentence.
    def __init__(self, d_model: int, seq_len: int, dropout: float):
        # seq_len -> maximum input length of the sequence
        # dropout -> to make the model less overfit
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        self.seq_len = seq_len

        # Create a matrix of shape (seq_len , d_model)
        pe = torch.zeros(seq_len, d_model)
        # Create a vector of shape (seq_len, 1)
        position = torch.arange(0,seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))
        # Applying sin to even positions and cos to odd positions
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0) #(1, seq_len, d_model) as we have batches
            
        '''buffers are tensors that are saved in the model's state but are not updated during training
        (i.e., they are not considered model parameters and do not have gradients).'''
        self.register_buffer('pe',pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad(False) #As we don't want the positional encodings to change during training.
        return self.dropout(x)
        
class LayerNormalization(nn.Module):
    def __init__(self, eps: float = 10**-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) # Multiplied
        self.beta = nn.Parameter(torch.zeros(1)) # Added
    
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim = True)
        std = x.std(dim=-1, keepdim = True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias
    
class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.layer1 = nn.Linear(d_model, d_ff)
        self.droput = nn.Dropout(dropout)
        self.layer2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.layer2(self.droput(torch.relu(self.layer1(x))))
    
class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model is not divisible by h"

        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]

        #(B, h, seq, d_k) @ (B, h, d_k, seq) -> (B, h, seq, seq)
        attention_scores = (query @ key.transpose(-2,-1)) / math.sqrt(d_k)

        if mask is not None:
            attention_scores.masked_fill(mask==0, -1e9)
        attention_scores = attention_scores.softmax(dim = -1) #(B, h, seq, seq)

        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        query = self.w_q(q) # (Batch, seq_len, d_model) -> (Batch, seq_len, d_model)
        key = self.w_k(k)
        value = self.w_v(v)

        # (B, seq, d_model) -> (B, seq, h, d_k) -> (B, h, seq, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1,2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1,2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1,2)

        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        x = x.transpose(1,2) # (B, h, seq, d_k) -> (B, seq, h, d_k)
        x = x.contiguous().view(x.shape[0], -1, self.h*self.d_k) # (B, seq, h, d_k) -> (B, seq, d_model)

        return self.w_o(x) #(B, seq, d_model)
    
class ResidualConnection(nn.Module):
    def __init__(self, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        return x+ self.dropout(sublayer(self.norm(x)))
    
class EncoderBlock(nn.Module):
    def __init__(self, self_attn_block: MultiHeadAttentionBlock, feed_forward_block : FeedForwardBlock, dropout: float):
        super().__init__()
        self.self_attn_block = self_attn_block
        self.feed_forward_block = feed_forward_block
        self.residual_connection = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self,x, src_mask):
        x = self.residual_connection[0](x, lambda x: self.self_attn_block(x,x,x, src_mask))
        x = self.residual_connection[1](x, self.feed_forward_block)
        return x
    
class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class DecoderBlock(nn.Module):
    def __init__(self, self_attn_block: MultiHeadAttentionBlock, cross_attn_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float):
        super().__init__()
        self.self_attn_block = self_attn_block
        self.cross_attn_block = cross_attn_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attn_block(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attn_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)

class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)
    
class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        #(B, seq, d_model) -> (B, seq, vocab_size)
        return torch.log_softmax(self.proj(x), dim = -1)
    
class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbedding, tgt_embed: InputEmbedding,
                src_pos: PositionEncoding, tgt_pos: PositionEncoding, projection_layer: ProjectionLayer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.projection_layer = projection_layer

    def encoder(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    
    def decoder(self, encoder_output, src_mask, tgt, tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
    def project(self, x):
        return self.projection_layer(x)


def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int,
                    d_model: int = 512, d_ff: int = 2048, N: int = 6, h: int = 8, dropout: float = 0.1):
    # Create embedding layers
    src_embed = InputEmbedding(d_model, src_vocab_size)
    tgt_embed = InputEmbedding(d_model, tgt_vocab_size)

    src_pos = PositionEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionEncoding(d_model, tgt_seq_len, dropout)

    # Create the position encoding layers
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attn_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attn_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # Create the decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attn_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attn_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(decoder_self_attn_block, decoder_cross_attn_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)

    # Create the encoder and the decoder
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    # Create the projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    # Create the transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)

    # Initialising the parameters to make the training faster, so that they do not get initialised with random values
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)

    return transformer