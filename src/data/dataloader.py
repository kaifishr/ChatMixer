import os
import re
import zipfile
import pathlib

import numpy
import random
import tqdm
import torch
import torchtext
from torch.utils.data import DataLoader

from src.config.config import Config
from src.data.dataset import CharDataset


def seed_worker(worker_id):
    """Seed dataloader workers."""
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)


def get_dataloader(config: Config) -> tuple[DataLoader, DataLoader]:
    """Creates dataloader for specified dataset."""

    dataset = config.dataloader.dataset
    num_workers = config.dataloader.num_workers
    batch_size = config.trainer.batch_size
    input_sequence_length = config.model.input_sequence_length
    output_sequence_length = config.model.output_sequence_length

    if dataset == "shakespeare":

        # Create folder for data.
        data_dir = "data/shakespeare/"
        pathlib.Path(data_dir).mkdir(parents=True, exist_ok=True)

        # Download data if not already done.
        dataset_url = "https://ocw.mit.edu/ans7870/6/6.006/s08/lecturenotes/files/t8.shakespeare.txt"
        torchtext.utils.download_from_url(url=dataset_url, root=data_dir)

        data_path = data_dir + "/t8.shakespeare.txt"
        with open(data_path, mode="r") as file:
            data = file.read()

        train_dataset = CharDataset(
            data=data, 
            input_length=input_sequence_length, 
            output_length=output_sequence_length
        )
        test_dataset = train_dataset

        config.data.num_classes = train_dataset.num_tokens
        config.data.num_tokens = train_dataset.num_tokens

    elif dataset == "lexicap":

        data = load_lexicap()

        train_dataset = CharDataset(
            data=data, 
            input_length=input_sequence_length, 
            output_length=output_sequence_length
        )
        test_dataset = train_dataset

        config.data.num_classes = train_dataset.num_tokens
        config.data.num_tokens = train_dataset.num_tokens

    else:
        raise NotImplementedError(f"Dataloader for {dataset} not implemented.")

    generator = torch.Generator()
    generator.manual_seed(config.random_seed)

    if "cuda" in str(config.trainer.device):
        pin_memory = True
    else:
        pin_memory = False

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=generator,
        shuffle=True,
        pin_memory=pin_memory,
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=2 * batch_size,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=generator,
        shuffle=False,
        pin_memory=pin_memory,
    )

    return train_loader, test_loader


def load_lexicap() -> str:
    """Downloads and cleans transcripts from Lex Fridman episodes.

    Script removes time stamps and merges all transcript into a (currently) ~30MB file.

    Transcripts can be found here: https://karpathy.ai/lexicap/
    """

    # Create folder for data.
    data_dir = "data/lexicap/"
    pathlib.Path(data_dir).mkdir(parents=True, exist_ok=True)

    # Download data if not already done.
    dataset_url = "https://karpathy.ai/lexicap/data.zip"
    torchtext.utils.download_from_url(url=dataset_url, root=data_dir)

    # Define regular expression pattern to remove time stamps.
    pattern = r"(\s)?(\d{1,2}:)?\d{2}:\d{2}.\d{3} --> (\d{1,2}:)?\d{2}:\d{2}.\d{3}"

    # Compile the regular expression
    regex = re.compile(pattern)

    transcripts = []

    cwd = os.getcwd()

    with zipfile.ZipFile(cwd + "/" + data_dir + "data.zip", mode="r") as zip_file:
        for name in tqdm.tqdm(zip_file.namelist(), desc="Cleaning"):
            # There are "small" and "large" files
            # for every transcript. Here we go with "large".
            if name.endswith("large.vtt"):
                with zip_file.open(name, mode="r") as file:
                    # Skip the header.
                    file.readline()
                    # Encode data.
                    data = str(file.read(), encoding="utf-8")
                    # Remove new lines.
                    data = " ".join(data.split())
                    # Remove time stamps with pattern defined above.
                    data = regex.sub("", data)
                    transcripts.append(data)

    transcripts = " ".join(transcripts)

    return transcripts
