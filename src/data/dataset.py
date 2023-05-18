"""Dataset class for character-level language models."""
import numpy
import torch
from torch.utils.data import Dataset


class CharDataset(Dataset):
    """Character-level dataset.

    Generates encoded batches of character sequences.

    Attributes:
        data: Data string.
        config: Configuration.
        input_length: Length of input sequence.
        output_length: Length of output sequence.
        char_to_index: Look table mapping character tokens to indices.
        index_to_char: Look table mapping indices to character tokens .
        num_tokens: Total number of tokens.
    """

    def __init__(self, data: str, input_length: int = 1, output_length: int = 1):
        self.data = data
        self.input_sequence_length = input_length
        self.output_sequence_length = output_length

        # Create lookup-tables with character-index-pairs in both directions.
        chars = sorted(list(set(self.data)))
        self.char_to_index = {char: i for i, char in enumerate(chars)}
        self.index_to_char = {i: char for i, char in enumerate(chars)}

        self.num_tokens = len(chars)

        print(f"Number of characters: {len(data)/1e6:.3f} M\n")
        print(f"Unique characters: {self.num_tokens}\n")

        self._length = len(self.data) - (
            self.input_sequence_length + self.output_sequence_length
        )

    def __len__(self):
        return self._length

    def _get_sequences(self, idx: int) -> tuple[list[int], list[int]]:
        """Creates a connected sequence.

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

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Gets sequence of characters from data.

        Extracts sequence of characters from data using defined method.

        Args:
            idx: Index to access string stored in data.

        Returns:
            Input and target tensors.
        """
        idx_sequence_x, idx_sequence_y = self._get_sequences(idx=idx)
        x = torch.tensor(data=idx_sequence_x, dtype=torch.long)
        y = torch.tensor(data=idx_sequence_y, dtype=torch.long)
        return x, y


class MaskedCharDataset(Dataset):
    """Masked character-level dataset.

    Generates encoded batches of character sequences. For 'output_length = 1'
    this class behaves like 'CharDataset'.

    Attributes:
        data: Data string.
        config: Configuration.
        input_length: Length of input sequence.
        output_length: Length of output sequence.
        char_to_index: Look table mapping character tokens to indices.
        index_to_char: Look table mapping indices to character tokens .
        num_tokens: Total number of tokens.
    """

    def __init__(self, data: str, input_length: int = 1, output_length: int = 1):
        assert output_length < input_length

        self.data = data
        self.input_sequence_length = input_length
        self.output_sequence_length = output_length

        self.mask_char = "⁂"
        self.data += self.mask_char

        # Create lookup-tables with character-index-pairs in both directions.
        chars = sorted(list(set(self.data)))
        self.char_to_index = {char: i for i, char in enumerate(chars)}
        self.index_to_char = {i: char for i, char in enumerate(chars)}

        self.num_tokens = len(chars)

        print(f"Number of characters: {len(data)/1e6:.3f} M\n")
        print(f"Unique characters: {self.num_tokens}\n")

        self._length = len(self.data) - (self.input_sequence_length + 1)

    def __len__(self):
        return self._length

    def _get_sequences(self, idx: int) -> tuple[list[int], list[int]]:
        """Creates a masked sequence.

        Creates a sequence of randomly masked inputs. Targets are the masked
        input tokens as well as the next token in the sequence. The idea is to
        force the model to interpolate (predict missing / masked tokens in the
        sequence) as well as to extrapolate (predict next token at end of the
        sequence).

        From a sequence such as

            char_sequence = "The quick brown fox"

        the following input and target sequences are created:

        char_sequence_x = "Th* qu*ck b*own*fo"
        char_sequence_y = "eir x"

        Where the '*' marks the masked input characters.

        Args:
            idx: Index to access string stored in 'data'

        Returns:
            Input and target tensors.
        """
        # Choose ranomd number of tokens to be dropped.
        idx_drop = numpy.random.choice(
            a=self.input_sequence_length,
            size=self.output_sequence_length - 1,
            replace=False,
        )
        idx_drop = numpy.sort(idx_drop)

        # Extract sequence at random position 'idx'
        sequence_length = self.input_sequence_length + 1
        char_sequence = self.data[idx : idx + sequence_length]

        # Translate tokens to indices
        idx_sequence = [self.char_to_index[char] for char in char_sequence]

        # Create masked input sequence.
        idx_sequence_x = numpy.array(idx_sequence[:-1])
        idx_sequence_x[idx_drop] = self.char_to_index[self.mask_char]

        # Create output sequence with masked target tokens and next token.
        idx_sequence_y = numpy.array(self.output_sequence_length * [0])
        idx_sequence_y[:-1] = numpy.array(idx_sequence)[idx_drop]
        idx_sequence_y[-1] = idx_sequence[-1]

        return idx_sequence_x, idx_sequence_y

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Gets sequence of characters from data.

        Extracts sequence of characters from data using defined method.

        Args:
            idx: Index to access string stored in data.

        Returns:
            Input and target tensors.
        """
        idx_sequence_x, idx_sequence_y = self._get_sequences(idx=idx)
        x = torch.tensor(data=idx_sequence_x, dtype=torch.long)
        y = torch.tensor(data=idx_sequence_y, dtype=torch.long)
        return x, y
