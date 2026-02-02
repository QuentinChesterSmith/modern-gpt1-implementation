import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model import GPT
from dataset import GutenbergDataset


# Lookup Tables

OPTIMIZERS = {
    "Adam": optim.Adam,
    "SGD": optim.SGD
}
LOSS_FUNCTIONS = {
    "CrossEntropyLoss": nn.CrossEntropyLoss,
}


def train(config):
    # Load Configs

    # Hyperparameters
    epochs = config["hyperparameters"]["epochs"]
    seq_length = config["hyperparameters"]["seq_len"]
    batch_size = config["hyperparameters"]["batch_size"]

    # Model Architecture
    num_layers = config["model"]["num_layers"]
    embed_dim = config["model"]["embed_dim"]
    num_heads = config["model"]["num_heads"]
    vocab_size = config["model"]["vocab_size"]

    # Optimization & Loss
    optimizer_class = OPTIMIZERS[config["optimization"]["optimizer"]]
    lr = config["optimization"]["lr"]
    loss_func_class = LOSS_FUNCTIONS[config["loss"]["loss_func"]]

    # Logging Freq
    train_loss_freq = config["logging_freq"]["train_loss"]

    train_set = GutenbergDataset(seq_length)
    train_dl = DataLoader(train_set, batch_size)

    model = GPT(num_layers, embed_dim, num_heads, vocab_size)

    optimizer = optimizer_class(model.parameters(), lr)
    loss_func = loss_func_class()

    model.train()
    steps = 0
    running_loss = 0
    print("Begining Training")
    for epoch in range(epochs):
        for (X, y) in train_dl:
            optimizer.zero_grad()
            logits = model(X)
            
            # Flatten the batch and time dimension for loss calculations
            logits = logits.view(logits.shape[0]*logits.shape[1], logits.shape[2])
            y = y.view(y.shape[0]*y.shape[1])

            loss = loss_func(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            steps+=1

            # Logging
            if steps % train_loss_freq == 0:
                print(f"Step {steps} Training Loss: {running_loss/train_loss_freq}")
                running_loss = 0