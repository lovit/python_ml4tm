__description__ = 'PyTorch utilities for handling text'

# functions
from .data import load_name_data
from .data import load_name_data_as_dataset
from .preprocess import unicode_to_ascii
from .preprocess import ascii_to_onehot
from .preprocess import sequence_to_onehot
from .preprocess import ascii_to_index_seq

# values
from .preprocess import unicode_letter_padding_idx
from .preprocess import unicode_letters_dim
from .preprocess import unicode_letter_unknown_idx
from .utils import installpath
