import string
import torch
import unicodedata


all_unicode_letters = string.ascii_letters + " .,;'"
ascii_mapper = {c:idx for idx, c in enumerate(all_unicode_letters)}

n_unicode_letters = len(all_unicode_letters)
unicode_letters_dim = n_unicode_letters + 1 # last index for unknown
unicode_letter_unknown_idx = n_unicode_letters
unicode_letter_padding_idx = unicode_letters_dim + 1 # last index for padding


def unicode_to_ascii(s):
    """
    # 유니코드 문자열을 ASCII로 변환, https://stackoverflow.com/a/518232/2809427

    Usage
    -----
        >>> unicode_to_ascii('Ślusàrski')
        $ Slusarski
    """
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_unicode_letters
    )

def ascii_to_onehot(s, image_len=-1):
    """
    It transforms string to onehot image.

    Arguments
    ---------
    s : str
        which consists with ascii character
    image_len : int
        Maximum image length.
        If s is longer than image_len, it only consider first image_len characters
        If s is shorter than image_len, it fills the last parts with zero.
        If image_len is -1, it the length of image is same with length of s

    Returns
    -------
    image : torch.LongTensor
        (image_len, 58)
        Last column stands for unknown character

    Usage
    -----
    """
    if image_len > 0:
        s = s[:image_len]
    else:
        image_len = len(s)

    index_seq = ascii_to_index_seq(s, image_len=-1)
    image = sequence_to_onehot(index_seq, unicode_letters_dim, image_len)
    return image

def sequence_to_onehot(seq, dim, image_len=-1):
    """
    Usage
    -----
        >>> sequence_to_onehot([0, 1, 2], dim=5, image_len=4)
        $ tensor([[1, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0],
                  [0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 0]])
    """
    if image_len < 0:
        image_len = len(seq)
    image = torch.zeros(image_len, dim, dtype=torch.long)
    for i, j in enumerate(seq):
        image[i,j] = 1
    return image

def ascii_to_index_seq(s, image_len):
    """
    Usage
    -----
        >>> ascii_to_index_seq('abc')
        $ [0, 1, 2]

        >>> ascii_to_index_seq('abc', image_len=5)
        $ [0, 1, 2, 59, 59]
    """
    idx_seq = [ascii_mapper.get(c, n_unicode_letters) for c in s]
    if image_len > 0:
        idx_seq = idx_seq + [unicode_letter_padding_idx] * (image_len - len(idx_seq))
    return idx_seq
