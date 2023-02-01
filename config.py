import torch

# torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

hyperparams = dict(
    batch_size=64,  # number of parallel sequences
    block_size=512,  # context length for predictions
    n_embed=512,
    n_head=8,
    n_blocks=10,
    head_size=16,
    dropout=0.2
)
training = dict(
    max_iters=10000,
    eval_iters=200,
    eval_interval=500,
    learning_rate=1.5e-4
)
file = dict(
    path_to_dataset="F:/ml/gpt/dataset/ml-archive/ML_dataset_notime.txt",
    model_version=2,
    model_identifier="botnessa",
    path_to_models="./models/botnessa/",
    model_extension=".pt",
    load_type="fresh"  # type of loading we want, fresh, latest or a specific name
)


class DotDict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
