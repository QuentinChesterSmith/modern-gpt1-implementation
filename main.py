import yaml
from train import train

def main(config):
    train(config)

if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    main(config)