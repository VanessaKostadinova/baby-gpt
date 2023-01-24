import torch
import model
import os

# hyperparams
#torch.manual_seed(12345)
batch_size = 64  # number of parallel sequences
block_size = 256  # context length for predictions
max_iters = 5000
eval_iters = 200
eval_interval = 500
learning_rate = 3e-4
dev = "cuda:0" if torch.cuda.is_available() else "cpu"
device = torch.device(dev)
path_to_dataset = "F:/ml/gpt/dataset/tiny-shakespeare.txt"
model_version = 1
model_identifier = "shakespeare"
path_to_models = "./models/"
model_extension = ".pt"
# type of loading we want, fresh, latest or a specific name
load_type = "latest"

# --------------

# finding model we will train
if load_type == "latest":
    load_model = sorted(os.listdir(path_to_models))[-1]
    load_step = int(load_model[:-len(model_extension)].split("_")[-1])
    print(f"training model {load_model}")
elif load_type != "fresh":
    load_model = load_type
    load_step = int(load_model[:-len(model_extension)].split("_")[-1])
    print(f"training model {load_model}")
else:
    load_step = 0
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


m = model.GPTModel().to(device)

if load_type != "fresh":
    m.load_state_dict(torch.load(f"{path_to_models}{load_model}"))

m.train()
optimiser = torch.optim.AdamW(m.parameters(), lr=learning_rate)

for step in range(load_step, max_iters):
    if step % eval_interval == 0:
        losses = estimate_loss()
        print(f"saving model_{model_identifier}_{model_version:02}_step_{step}")
        torch.save(
            m.state_dict(),
            f"{path_to_models}model_{model_identifier}_{model_version:02}_step_{step}{model_extension}"
        )
        print(f"step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch("train")

    # eval loss
    logits, loss = m(xb, yb)
    optimiser.zero_grad(set_to_none=True)
    loss.backward()
    optimiser.step()

idx = torch.zeros((1, 1), dtype=torch.long).to(device)
print(decode(m.generate(idx, 10000)[0].tolist()))
