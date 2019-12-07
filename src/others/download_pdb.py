#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# file_name : download_pdb.py
# time      : 3/22/2019 13:54
# author    : ruiyang
# email     : ww_sry@163.com
# ------------------------------

import sys
import pandas as pd
import os
def download_pdb(name_list,outpath):
    """
    :function: download pdb archive by pdbid from server: https://files.rcsb.org/download/
    :param name_list: 存储了pdbid的python list, set, numpy_array等可迭代对象
    :param outpath: 文件存储路径
    :return: none
    """
    if not os.path.exists(outpath):
        os.mkdir(outpath)
    os.chdir(outpath)
    print('len of namelist:', len(name_list))
    errorlist = []
    for pdbid in name_list:
        print(pdbid)
        print('download begin')
        try:
            os.system('wget https://files.rcsb.org/download/' + pdbid[:4] + '.pdb')
        except:
            errorlist.append(pdbid)
    print('len of errorlist:',len(errorlist))

    return(errorlist)


if __name__ == '__main__':
    dataset_name = sys.argv[1]
    csv_path = r'../datasets/%s/%s_new.csv'%(dataset_name,dataset_name)
    outpath = r'../datasets/%s/pdb%s'%(dataset_name,dataset_name)
    f = open(csv_path,'r')
    mutation_df = pd.read_csv(f)
    f.close()
    pdbid_array = mutation_df.loc[:,'PDB'].values
    pdbid_array = set(pdbid_array)
    download_pdb(pdbid_array, outpath)