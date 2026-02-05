import torch, yaml
import tiktoken
from torch.nn.functional import pad, softmax
from model import GPT

def generate_token(model, tokens):
    logits = model(torch.tensor(tokens))
    output_probs = softmax(logits, dim=1)
    return torch.multinomial(output_probs, num_samples=1)


def generate_text(model, n_tokens, text):
    model.eval()
    encoder = tiktoken.get_encoding("gpt2")
    tokens = encoder.encode(text)
    for i in range(n_tokens):
        tokens.append(list(generate_token(model, tokens)))
    
    return encoder.decode(tokens)

if __name__ == "__main__":

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Model Architecture
    num_layers = config["model"]["num_layers"]
    embed_dim = config["model"]["embed_dim"]
    num_heads = config["model"]["num_heads"]
    vocab_size = config["model"]["vocab_size"]

    while True:
        model_checkpoint = input("Model Checkpoint: ")
        length = int(input("How many tokens to generate?"))
        text = input("Text: ")

        model = GPT(num_layers, embed_dim, num_heads, vocab_size)
        model.load_state_dict(torch.load(model_checkpoint))
        generate_text(model, n_tokens=length, text=text)