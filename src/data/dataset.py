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

    def _get_sequence_connected(self, idx: int) -> tuple[list[int], list[int]]:
        """Creates connected sequence.

        For 'data' holding a sequence of characters like

        data = "The quick brown Fox jumps"

        with idx = 4, input_sequence_length = 8, and
        output_sequence_length = 2, the following sequence
        of characters are extracted from the data
        sequence

        char_sequence = "quick brow"

        which is being encoded as a list of integers

        encoded_sequence = [9, 1, 4, 8, 2, 5, 3, 7, 6, 0]
                            q  u  i  c  k " " b  r  o, w

        From this list, the following input and target is created:

        x = [9, 1, 4, 8, 2, 5, 3, 7]
        y = [6, 0]
        
        Args:
            idx: Index to access string stored in 'data'

        Returns:
            Input and target tensors.
        """
        sequence_length = self.input_sequence_length + self.output_sequence_length
        char_sequence = self.data[idx : idx + sequence_length]
        idx_sequence = [self.char_to_index[char] for char in char_sequence]
        idx_sequence_x = idx_sequence[: self.input_sequence_length]
        idx_sequence_y = idx_sequence[self.input_sequence_length :]
        return idx_sequence_x, idx_sequence_y

    def _get_sequence_displaced(self, idx: int) -> tuple[list[int], list[int]]:
        """Creates interleaved sequence.

        Creates a sequence of interleaved inputs and targets that forces the
        model to interpolate (predict missing tokens in the sequence) as 
        well as extrapolate (predict token at end of the sequence). Here we
        use slicing operations '[-2::-2][::-1]' for the input and 
        '[-1::-2][::-1]' for the output to ensure that the to predicted output 
        sequence always contains the sequence's last character.

        TODO: Not very elegant. Model learns that always every second letter 
        is missing. Try with random input-target pair.
        TODO: Add explanation like above.
        char_sequence = "quick brow"
        char_sequence_input = "qikbo"
        char_sequence_output = "uc rw"
        
        Args:
            idx: Index to access string stored in 'data'

        Returns:
            Input and target tensors.
        """
        assert self.input_sequence_length % 2 == 0 
        assert self.input_sequence_length == self.output_sequence_length
        sequence_length = self.input_sequence_length
        char_sequence = self.data[idx : idx + 2 * sequence_length]
        idx_sequence = [self.char_to_index[char] for char in char_sequence]
        idx_sequence_x = idx_sequence[-2::-2][::-1]
        idx_sequence_y = idx_sequence[-1::-2][::-1]
        return idx_sequence_x, idx_sequence_y

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Gets sequence of characters from data.

        Extracts sequence of characters from data using defined method.

        Args:
            idx: Index to access string stored in data.

        Returns:
            Input and target tensors.
        """
        # idx_sequence_x, idx_sequence_y = self._get_sequence_connected(idx=idx)
        idx_sequence_x, idx_sequence_y = self._get_sequence_displaced(idx=idx)
        x = torch.tensor(data=idx_sequence_x, dtype=torch.long)
        y = torch.tensor(data=idx_sequence_y, dtype=torch.long)
        return x, y

    # def __getitem__(self, idx):
    #     """Extracts sequence of characters from data.

    #     For 'data' holding a sequence of characters like

    #     data = "The quick brown Fox jumps"

    #     with idx = 4, input_sequence_length = 8, and
    #     output_sequence_length = 2, the following sequence
    #     of characters are extracted from the data
    #     sequence

    #     char_sequence = "quick brow"

    #     which is being encoded as a list of integers

    #     encoded_sequence = [9, 1, 4, 8, 2, 5, 3, 7, 6, 0]
    #                         q  u  i  c  k " " b  r  o, w

    #     From this list, the following input and target is created:

    #     x = [9, 1, 4, 8, 2, 5, 3, 7]
    #     y = [6, 0]

    #     Args:
    #         idx: Index to access string stored in data.
    #     """
    #     sequence_length = self.input_sequence_length + self.output_sequence_length
    #     char_sequence = self.data[idx : idx + sequence_length]
    #     int_sequence = [self.char_to_index[char] for char in char_sequence]
    #     x = torch.tensor(data=int_sequence[: self.input_sequence_length], dtype=torch.long)
    #     y = torch.tensor(data=int_sequence[self.input_sequence_length :], dtype=torch.long)
    #     return x, y
