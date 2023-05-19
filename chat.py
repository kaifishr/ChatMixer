"""Methods to chat with pre-trained language network."""
import argparse
from argparse import Namespace

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

from src.config.config import Config
from src.config.config import init_config
from src.data.dataloader import get_dataloader
from src.modules.model import MLPMixer
from src.modules.model import ConvMixer
from src.modules.model import ConvModel
from src.utils.tools import load_checkpoint


class Chat:
    """Chat class.

    Uses a autoregressive model to generate text provided a prompt.

    Attributes:
        model: An autoregressive model.
        dataset: Dataset model has been trained on.
        config: Configuration.
        valid_characters: Legal characters.

    """

    def __init__(self, model: torch.nn.Module, dataset: Dataset, config: Config, args: Namespace):
        """Initializes chat class."""
        self.model = model
        self.dataset = dataset
        self.config = config

        self.valid_characters = list(self.dataset.char_to_index)

        self.device = self.config.trainer.device
        self.input_sequence_length = self.config.model.input_sequence_length

        # Maximum number of generated tokens.
        self.num_tokens = args.num_tokens
        self.temperature = args.temperature
        self.do_sample = args.do_sample
        # self.top_k = 10

        # NOTE: Depending on which dataset is used for training, this variable
        # may vary. For 'MaskedCharDataset' use 'idx_max_confidence = -1'.
        self.idx_max_confidence = 0

    @torch.no_grad()
    def _generate(self, prompt: str) -> str:
        """Generates text from prompt.

        Args:
            input_text: Prompt text.

        Returns:
            Text generated by model.
        """
        # Encode input characters as integer using lookup table from dataloader.
        data = [self.dataset.char_to_index[char] for char in prompt]

        # Create input tensor from encoded characters.
        x = torch.tensor(data=data, dtype=torch.long)[None, ...].to(self.device)

        # Generate some tokens
        for _ in range(self.num_tokens):
            # Make sure that the sequence length is smaller than max sequence length.
            sequence = (
                x
                if x.size(-1) <= self.input_sequence_length
                else x[:, -self.input_sequence_length :]
            )

            # Feed sequence into model.
            logits = self.model(sequence)

            # High temperature: make model more creative (text generation).
            # Low temperature: make model more confident (knowledge retrieval).
            # Take first prediction (0) as it is probably associated with the
            # highest confidence.
            logits = logits[:, self.idx_max_confidence, :] / self.temperature

            # Convert logits to probabilities.
            probabilities = F.softmax(input=logits, dim=-1)

            if self.do_sample:
                index_next_token = torch.multinomial(probabilities, num_samples=1)
            else:
                # Take the most likely next token.
                _, index_next_token = torch.topk(probabilities, k=1, dim=-1)

            # Add index of most likely token to running sequence.
            x = torch.cat((x, index_next_token), dim=-1)

        # Remove prompt from sequence:
        x = x[:, len(prompt) :]

        output = "".join([self.dataset.index_to_char[int(index)] for index in x[0]])

        return output

    def _is_valid_prompt(self, prompt: str) -> bool:
        """Checks if input prompt contains any illegal characters."""
        for character in prompt:
            if character not in self.valid_characters:
                print(f"\nCharacter '{character}' was not part of the training data.")
                return False
        return True

    def _add_padding(self, prompt: str, char: str = " ") -> str:
        """Pads input prompt to have correct size."""
        return prompt.rjust(self.input_sequence_length, " ")

    def test(self):
        """Tests model with some simple prompts."""

        prompts = [
            "Why is there something rather than nothing?",
            "What is the meaning of life?",
            "Alice was so tired when she got home so she went",
            "Jack wanted to read a book, so he went to",
            "If I throw a ball up in the air, eventually, it will",
        ]

        for prompt in prompts:
            print(f"\n[User]\n{prompt}\n")
            if self._is_valid_prompt(prompt=prompt):
                prompt = self._add_padding(prompt=prompt)
                output = self._generate(prompt=prompt)
                print(f"\n[ChatMixer]\n{output}\n")

    def run(self):
        """Runs chat."""
        is_running = True

        print("\nPlease enter a prompt.\n")

        while is_running:
            print("[User]")
            prompt = input()

            if prompt.startswith("!"):
                command = prompt[1:]
                if command == "exit":
                    is_running = False
                elif command.startswith("--"):
                    argument, value = command[2:].split(" ")
                    if argument == "temp":
                        self.temperature = float(value)
                    elif argument == "num-tokens":
                        self.num_tokens = int(value)
                    elif argument == "do-sample":
                        self.do_sample = bool(value)
                    else:
                        print(f"Argument '{argument}' not recognized.")
                else:
                    print(f"Command '{command}' not recognized.")
                continue
            elif prompt == "":
                continue

            # Feed text to model
            if is_running and self._is_valid_prompt(prompt=prompt):
                prompt = self._add_padding(prompt=prompt)
                output = self._generate(prompt=prompt)
                print(f"\n[ChatMixer]\n{output}\n")

        print("Bye!")


def argument_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        prog="ChatMixer",
        description="A simple program to interact with a language model.",
    )

    parser.add_argument(
        "--num-tokens",
        dest="num_tokens",
        help="Number of tokens to be generated.",
        default=100,
        type=int,
    )

    parser.add_argument(
        "--temp",
        dest="temperature",
        help="Creativity parameter.",
        default=0.6,
        type=float,
    )
    parser.add_argument(
        "--do-sample",
        action="store_true",
        dest="do_sample",
        help="Sample from a multinomial distribution.",
    )

    return parser.parse_args()


if __name__ == "__main__":

    args = argument_parser()

    # Get configuration file
    config_path = "config.yml"
    config = init_config(file_path=config_path)

    # Load dataloader (also initializes config).
    dataloader, _ = get_dataloader(config=config)

    # Get dataset with encoder-decoder methods.
    dataset = dataloader.dataset

    # Get the model
    model_type = config.model.type
    if model_type == "mlpmixer":
        model = MLPMixer(config=config)
    elif model_type == "convmixer":
        model = ConvMixer(config=config)
    elif model_type == "cnn":
        model = ConvModel(config=config)
    else:
        raise NotImplementedError(f"Model type {model_type} not available.")
    model = torch.jit.script(model)

    ckpt_dir = config.dirs.weights
    model_name = config.load_model.model_name
    load_checkpoint(model=model, ckpt_dir=ckpt_dir, model_name=model_name)
    model.to(config.trainer.device)
    model.eval()

    chat = Chat(model=model, dataset=dataset, config=config, args=args)
    # chat.test()
    chat.run()
