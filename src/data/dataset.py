"""Dataset class for character-level language models."""
import torch
from torch.utils.data import Dataset


class CharDataset(Dataset):
    """Character-level dataset.

    Generates encoded batches of character sequences.

    Attributes:
        data:
        config:
        input_length: Length of input sequence.
        output_length: Length of output sequence.
        char_to_index:
        index_to_char:
        num_chars:
    """

    def __init__(self, data: str, input_length: int = 1, output_length: int = 1):

        self.data = data

        self.input_sequence_length = input_length
        self.output_sequence_length = output_length

        chars = sorted(list(set(data)))

        # Create lookup-tables with character-index-pairs in both directions.
        self.char_to_index = {char: i for i, char in enumerate(chars)}
        self.index_to_char = {i: char for i, char in enumerate(chars)}

        self.num_tokens = len(chars)

        print(f"Number of characters: {len(data)/1e6:.3f} M\n")
        print(f"Unique characters: {self.num_tokens}\n")

    def __len__(self):
        return len(self.data) - (self.input_sequence_length + self.output_sequence_length)

    def __getitem__(self, idx):
        """Extracts sequence of characters from data.

        For data holding a sequence of characters

        data = "The quick brown Fox jumps"

        idx=4 and input_sequence_length=8, 
        output_sequence_length=2 the following block
        of characters are extracted from the data
        sequence

        char_block = "quick brow"

        which is being encoded as a list of integers:

        encoded_block = [9, 1, 4, 8, 2, 5, 3, 7, 6, 0]
                         q  u  i  c  k " " b  r  o, w

        From this list, the following input and target
        is created:

        x = [9, 1, 4, 8, 2, 5, 3, 7]
        y = [6, 0]

        Args:
            idx: Index to access string stored in data.
        """
        sequence_length = self.input_sequence_length + self.output_sequence_length
        char_sequence = self.data[idx : idx + sequence_length]
        int_sequence = [self.char_to_index[char] for char in char_sequence]
        x = torch.tensor(data=int_sequence[:self.input_sequence_length], dtype=torch.long)
        y = torch.tensor(data=int_sequence[self.input_sequence_length:], dtype=torch.long)
        return x, y
