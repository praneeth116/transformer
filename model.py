import torch
import torch.nn as nn
import math

d_model = 10
div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))
print(div_term)

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