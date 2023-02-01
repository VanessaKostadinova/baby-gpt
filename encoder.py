import tiktoken


# set up simple encoder
def get_char_enc(data):
    chars = sorted(list(set(data)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
    return encode, decode, vocab_size
# --------------


def get_tik_token_enc():
    enc = tiktoken.get_encoding("gpt2")
    encode = enc.encode
    decode = enc.decode
    vocab_size = enc.n_vocab

    return encode, decode, vocab_size
