import torch
import model
import os
import encoder
import config

def set_globals(dict):
    for k in dict:
        globals()[k] = dict[k]

globals()["device"] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

set_globals(config.file)

# finding model we will train
# get the latest model for specified file based on step
if load_type == "latest":
    model_prefix = f"model_{model_identifier}_{model_version:02}_step_"

    valid_files = [x for x in os.listdir(path_to_models)
                   if x.startswith(model_prefix)]

    latest_step = int(sorted(name[:-len(model_extension)].split("_")[-1] for name in valid_files)[-1])
    model_name = f"{model_prefix}{latest_step}{model_extension}"
    print(f"training model {model_name}")
    load_data = torch.load(f"{path_to_models}{model_name}")
    hyperparams = load_data["hyperparams"]
# if we don't want a fresh model we've probably specified a file
elif load_type != "fresh":
    model_name = load_type
    model_prefix = f"model_{model_identifier}_{model_version:02}_step_"
    latest_step = int(model_name[:-len(model_extension)].split("_")[-1])
    print(f"training model {model_name}")
    load_data = torch.load(f"{path_to_models}{model_name}")
    hyperparams = load_data["hyperparams"]
else:
    model_prefix = f"model_{model_identifier}_{model_version:02}_step_"
    latest_step = 0
    hyperparams = config.hyperparams
    load_data = None
# --------------

set_globals(config.hyperparams)
set_globals(config.training)

# making encoder
with open(path_to_dataset, mode="r", encoding="utf-8") as file:
    text = file.read()

encode, decode, vocab_size = encoder.get_tik_token_enc()
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


def save_model(model, step):
    print(f"saving model_{model_identifier}_{model_version:02}_step_{step}")
    torch.save(
        dict(
            model=model.state_dict(),
            params=config.hyperparams
        ),
        f"{path_to_models}{model_prefix}{step}{model_extension}"
    )


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


m = model.GPTModel(vocab_size).to(device)

if load_data is not None:
    m.load_state_dict(load_data["model"])

m.train()
optimiser = torch.optim.AdamW(m.parameters(), lr=learning_rate)

print(f"model parameter count: {sum(p.numel() for p in m.parameters() if p.requires_grad):,}")

for step in range(latest_step, max_iters):
    if step % eval_interval == 0:
        losses = estimate_loss()
        save_model(m, step)
        print(f"step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch("train")

    # eval loss
    logits, loss = m(xb, yb)
    optimiser.zero_grad(set_to_none=True)
    loss.backward()
    optimiser.step()

save_model(m, max_iters)

idx = torch.zeros((1, 1), dtype=torch.long).to(device)
print(decode(m.generate(idx, 10000)[0].tolist()))
