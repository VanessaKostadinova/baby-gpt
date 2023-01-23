import tiktoken
import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparams
torch.manual_seed(12345)
batch_size = 64  # number of parallel sequences
block_size = 256  # context length for predictions
max_iters = 5000
eval_iters = 200
eval_interval = 500
learning_rate = 3e-4
dev = "cuda:0" if torch.cuda.is_available() else "cpu"
device = torch.device(dev)
path_to_dataset = "F:/ml/gpt/dataset/tiny-shakespeare.txt"
n_embed = 384
n_head = 6
n_blocks = 6
head_size = 16
dropout = 0.2
# --------------

with open(path_to_dataset, mode="r") as file:
    text = file.read()

# set up simple encoder
chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])
# --------------


# train/val split
enc_data = torch.tensor(encode(text), dtype=torch.long, device=device)
n = int(0.9 * len(enc_data))
train_data = enc_data[:n]
val_data = enc_data[n:]


# --------------

# data loading
def get_batch(split):
    data = train_data if split == "train" else val_data
    # generate batch size number of random offsets, our context begins there
    ix = torch.randint((len(data) - block_size), (batch_size,))
    # x = context, a stack of block_size data inputs
    x = torch.stack([data[i:i + block_size] for i in ix]).to(device)
    # y = target, x but shifted by 1
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix]).to(device)
    return x, y


# --------------

@torch.no_grad()
def estimate_loss():
    out = {}
    m.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = m(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    m.train()
    return out


class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

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
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

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
            nn.Dropout(dropout)
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


class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(*[Block(n_embed, n_head) for _ in range(n_blocks)])
        self.l_norm = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

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
            idx_ctx = idx[:, -block_size:]
            # prediction
            logits, loss = self(idx_ctx)
            # get last timestep logits
            logits = logits[:, -1, :]  # (B, C)
            # softmax
            probs = F.softmax(logits, dim=1)  # (B, C)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


m = BigramLanguageModel().to(device)
optimiser = torch.optim.AdamW(m.parameters(), lr=learning_rate)

for step in range(max_iters):
    if step % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch("train")

    # eval loss
    logits, loss = m(xb, yb)
    optimiser.zero_grad(set_to_none=True)
    loss.backward()
    optimiser.step()

idx = torch.zeros((1, 1), dtype=torch.long).to(device)
print(decode(m.generate(idx, 10000)[0].tolist()))
