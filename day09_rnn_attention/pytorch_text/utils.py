import os
import numpy as np
from time import time, strftime, gmtime


installpath = os.path.dirname(__file__)

def time_from_t(base):
    t = time() - base
    return strftime("%H:%M:%S", gmtime(t))

def random_select(namedata):
    """
    namedata : [(category, [name, name, ... ]), 
                (category, [name, name, ... ]).
                ... ]
    return : name, category
    """
    category, names = namedata[np.random.randint(len(namedata))]
    name = names[np.random.randint(len(names))]
    return name, category
