#!/usr/bin/env python
'''
根据[~/mCNN/dataset/test1/feature/mCNN/wild/csv/3DVI_H_A_92_D/{center_CA.csv | center_CA_neighbor_._center_coord.npy}],
计算一些附加特征并拼接在 center_CA.csv 中,另存为 center_CA_appendix.csv
'''
import os,sys
import numpy as np
import pandas as pd

dataset_name = sys.argv[1]
csv_wild_dir = '/public/home/sry/mCNN/dataset/%s/feature/mCNN/wild/csv'%dataset_name
csv_mutant_dir = '/public/home/sry/mCNN/dataset/%s/feature/mCNN/mutant/csv'%dataset_name

def wild_main():
    # qsub loop
    for mutant_tag in os.listdir(csv_wild_dir):
        csv_pth = csv_wild_dir + '/' + mutant_tag + '/center_CA.csv'
        print(mutant_tag)

def mutant_main():
    # qsub loop
    pass

if __name__ == '__main__':
    wild_main()