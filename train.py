"""Main script to run experiments."""
from src.config.config import init_config
from src.data.dataloader import get_dataloader
from src.modules.model import MLPMixer
from src.modules.model import ConvMixer
from src.modules.model import ConvModel
from src.trainer.trainer import Trainer
from src.utils.tools import set_random_seed
from src.utils.tools import load_checkpoint
from src.utils.tools import count_model_parameters


def train_mixer():
    # Get configuration file.
    config = init_config(file_path="config.yml")

    # Seed random number generator.
    set_random_seed(seed=config.random_seed)

    # Get dataloader.
    dataloader = get_dataloader(config=config)

    # Get the model.
    model_type = config.model.type
    if model_type == "mlpmixer":
        model = MLPMixer(config=config)
    elif model_type == "convmixer":
        model = ConvMixer(config=config)
    elif model_type == "cnn":
        model = ConvModel(config=config)
    else:
        raise NotImplementedError(f"Model type {model_type} not available.")

    count_model_parameters(model=model)

    # Load pre-trained model.
    if config.load_model.is_activated:
        ckpt_dir = config.dirs.weights
        model_name = config.load_model.model_name
        load_checkpoint(model=model, ckpt_dir=ckpt_dir, model_name=model_name)

    model.to(config.trainer.device)

    print(config)
    trainer = Trainer(model=model, dataloader=dataloader, config=config)
    trainer.run()

    print("Experiment finished.")


if __name__ == "__main__":
    train_mixer()
