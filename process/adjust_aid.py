import tqdm, sklearn
import numpy as np
import os, time, sys
import pickle
from itertools import chain

if '../utils' not in sys.path:
    sys.path.append('../utils')

from data import Data

def main():
    data_bin, o_data_bin = '../run/seqVerbMC/data.bin', '../run/seqVerbMC/data.bin'
    data = Data()
    data.load(data_bin)
    data.id2arg = {y:x for x, y in data.arg_vocab.items()}
    data.save(o_data_bin)


if __name__ == "__main__":
    main()