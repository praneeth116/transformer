from typing import Any
import torch
import torch.nn as nn
from torch.utils.data import Dataset

class BilingualDataset(Dataset):
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):
        super().__init__()
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len

        self.sos_token = torch.tensor(tokenizer_src.token_to_id('[SOS]'), dtype=torch.int64)
        self.eos_token = torch.tensor(tokenizer_src.token_to_id('[EOS]'), dtype=torch.int64)
        self.pad_token = torch.tensor(tokenizer_src.token_to_id('[PAD]'), dtype=torch.int64)

    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, index: Any) -> Any:
        src_tgt_pair = self.ds[index]
        src_text = src_tgt_pair['translation'][self.src_lang]
        tgt_text = src_tgt_pair['translation'][self.tgt_lang]

        enc_input_tokens = self.tokenizer_src.encode(src_text).ids ## This gives input ids corresponding to each word.
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        enc_num_pad_tokens = self.seq_len - len(enc_input_tokens) - 2 ## Extra 2 for [SOS] and [EOS]
        dec_num_pad_tokens = self.seq_len - len(dec_input_tokens) - 1 ## Only '[SOS]' for decoder

        if enc_num_pad_tokens < 0 or dec_num_pad_tokens < 0:
            raise ValueError('Sentence is too long!')
        
        ## Add SOS, EOS and paddings to the source text
        encoder_input = torch.cat(
            [
                torch.tensor([self.sos_token]),
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                torch.tensor([self.eos_token]),
                torch.tensor([self.pad_token] * enc_num_pad_tokens, dtype=torch.int64)
            ]
        )

        ## Add SOS for decoder input.
        decoder_input = torch.cat(
            [
                torch.tensor([self.sos_token]),
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token] * dec_num_pad_tokens, dtype=torch.int64)
            ]
        )

        ## Add EOS to the label ( what we expect as output from the decoder )
        label = torch.cat(
            [
               torch.tensor(dec_input_tokens, dtype=torch.int64),
               torch.tensor([self.eos_token]),
               torch.tensor([self.pad_token] * dec_num_pad_tokens, dtype=torch.int64)
            ]
        )

        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "encoder_input": encoder_input, # (seq_len)
            "decoder_input": decoder_input, # (seq_len)
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), #(1,1,seq_len)
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(self.seq_len), #(1, 1, seq_len) & (1,seq_len, seq_len)
            "label": label,
            "src_text": src_text,
            "tgt_text": tgt_text
        }
    
def causal_mask(size):
    mask = torch.triu(torch.ones(1,size,size), diagonal=1).type(torch.int)
    return mask == 0