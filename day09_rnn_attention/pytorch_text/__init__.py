__description__ = 'PyTorch utilities for handling text'

# functions
from .data import load_name_data
from .data import load_name_data_as_dataset
from .model import RNNClassifier
from .preprocess import unicode_to_ascii
from .preprocess import ascii_to_onehot
from .preprocess import sequence_to_onehot
from .preprocess import ascii_to_index_seq
from .train import train_batch
from .utils import time_from_t
from .utils import random_select

# values
from .preprocess import unicode_letter_padding_idx
from .preprocess import unicode_letters_dim
from .preprocess import unicode_letter_unknown_idx
from .utils import installpath
