""""""
from src.modules.model import MlpMixer
from src.data.dataloader import get_dataloader
from src.config.config import init_config
from src.trainer.trainer import Trainer
from src.utils.tools import set_random_seed


def train_chat_mixer():

    config = init_config(file_path="config.yml")

    config.dataloader.dataset = "lexicap"

    set_random_seed(seed=config.random_seed)

    dataloader = get_dataloader(config=config)

    model = MLPMixer(config=config)
    model.to(config.trainer.device)

    print(config)
    trainer = Trainer(model=model, dataloader=dataloader, config=config)
    trainer.run()

    print("Experiment finished.")


def main():
    train_chat_mixer()


if __name__ == "__main__":
    main()