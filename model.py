import torch
import torch.nn as nn
from torch.nn import functional as F

from config import *

h = hyperparams

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class UnmaskedHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.key = nn.Linear(h.n_embed, h.head_size, bias=False)
        self.query = nn.Linear(h.n_embed, h.head_size, bias=False)
        self.value = nn.Linear(h.n_embed, h.head_size, bias=False)
        self.dropout = nn.Dropout(h.dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wgt = q @ k.transpose(-2, -1) * C ** -0.5
        wgt = F.softmax(wgt, dim=-1)
        wgt = self.dropout(wgt)
        v = self.value(x)
        out = wgt @ v
        return out


class MaskedHead(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(h.n_embed, head_size, bias=False)
        self.query = nn.Linear(h.n_embed, head_size, bias=False)
        self.value = nn.Linear(h.n_embed, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(h.block_size, h.block_size)))
        self.dropout = nn.Dropout(h.dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  # (B,T,16)
        q = self.query(x)  # (B,T,16)
        wgt = q @ k.transpose(-2, -1) * C ** -0.5  # (B, T, 16) @ (B, 16, T) -> (B, T, T)

        wgt = wgt.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # (B, T, T)
        wgt = F.softmax(wgt, dim=-1)
        wgt = self.dropout(wgt)
        v = self.value(x)
        out = wgt @ v
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([MaskedHead(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(h.n_embed, h.n_embed)
        self.dropout = nn.Dropout(h.dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(h.dropout)
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embed, n_head):
        super().__init__()
        head_size = n_embed // n_head
        self.sa_l_norm = nn.LayerNorm(n_embed)
        self.sa_heads = MultiHeadAttention(n_head, head_size)
        self.ffw_l_norm = nn.LayerNorm(n_embed)
        self.ffw = FeedForward(n_embed)

    def forward(self, x):
        x = x + self.sa_heads(self.sa_l_norm(x))  # (B, T, C)
        x = x + self.ffw(self.ffw_l_norm(x))  # (B, T, C)
        return x


class GPTModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        globals()["vocab_size"] = vocab_size
        self.token_embedding_table = nn.Embedding(vocab_size, h.n_embed)
        self.position_embedding_table = nn.Embedding(h.block_size, h.n_embed)
        self.blocks = nn.Sequential(*[Block(h.n_embed, h.n_head) for _ in range(h.n_blocks)])
        self.l_norm = nn.LayerNorm(h.n_embed)
        self.lm_head = nn.Linear(h.n_embed, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        # idx and targets are both (batch, time) tensors of int
        tok_emb = self.token_embedding_table(idx)  # (batch, time, channel)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T, C)
        x = tok_emb + pos_emb  # (B, T, C)
        x = self.blocks(x)
        x = self.l_norm(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        if targets == None:
            loss = None
        else:
            # need to reshape tensors because pytorch expects (B*T, C) in cross_entropy
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_ctx = idx[:, -h.block_size:]
            # prediction
            logits, loss = self(idx_ctx)
            # get last timestep logits
            logits = logits[:, -1, :]  # (B, C)
            # softmax
            probs = F.softmax(logits, dim=1)  # (B, C)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx
