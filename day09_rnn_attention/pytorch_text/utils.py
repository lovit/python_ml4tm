import os
from time import time, strftime, gmtime


installpath = os.path.dirname(__file__)

def time_from_t(base):
    t = time() - base
    return strftime("%H:%M:%S", gmtime(t))
