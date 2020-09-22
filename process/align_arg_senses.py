import tqdm, sklearn
import numpy as np
import os, time, sys
import pickle
from itertools import chain

if '../utils' not in sys.path:
    sys.path.append('../utils')

from data import Data

def main():
    data_bin, o_data_bin = '../run/seqVerbMC/data_subsrl_first_senses_argtrim.bin', '../run/seqVerbMC/data_subsrl_1sv_1sa_argtrim.bin'
    data = Data()
    data.load(data_bin)
    data.align_arg_senses_mix(filepath='/shared/muhao/wordnet/sense_wn_db.tsv', mode='first',limit=2)
    data.save(o_data_bin)


if __name__ == "__main__":
    main()