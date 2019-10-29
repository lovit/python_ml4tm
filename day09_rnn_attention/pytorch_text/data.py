import glob

from .preprocess import unicode_to_ascii
from .utils import installpath


def load_name_data(directory=None):
    """
    Returns
    -------
    rawdata : list of tuple
        [(category, [name, name, ... ]),
         (category, [name, name, ... ]),
         ...
        ]
    """
    if directory is None:
        directory = '{}/data/names/'.format(installpath)

    rawdata = []
    for path in glob.glob('{}/*.txt'.format(directory)):
        category = path.split('/')[-1][:-4]
        with open(path, encoding='utf-8') as f:
            names = [unicode_to_ascii(name.strip()) for name in f]
        rawdata.append((category, names))
    return rawdata
