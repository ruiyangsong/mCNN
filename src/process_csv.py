#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# file_name : process_csv.py
# time      : 3/08/2019 14:10
# author    : ruiyang
# email     : ww_sry@163.com

import sys
import numpy as np
import pandas as pd
from Bio.PDB.PDBParser import PDBParser


def split_csv(path):
    """
    :function: 将csv原文件分列，返回标准由DataFrame组成的csv,并加入主键列key.
    :param path: 描述突变信息的原始csv文件的绝对路径.
    :return: 分列后的df
    """

    file = open(path, 'r')
    data = pd.read_csv(file)
    file.close()
    col_index = data.columns[0].split(';')
    data_frame = data[data.columns[0]].str.split(';', n=-1, expand=True)
    data_frame.columns = col_index
    key = np.arange(len(data_frame)) #添加主键
    df_mutation = pd.DataFrame(
        {'key': key, 'PDB': data_frame.PDB, 'WILD_TYPE': data_frame.WILD_TYPE, 'CHAIN': data_frame.CHAIN,
         'POSITION': data_frame.POSITION, 'MUTANT': data_frame.MUTANT, 'PH': data_frame.PH,
         'TEMPERATURE': data_frame.TEMPERATURE, 'DDG': data_frame.DDG})
    return df_mutation

def process_csv(csv_path):
    """
    :function: 将 S1932.csv 处理成标准格式的 S1932_new.csv(加入主键列key,判断突变发生在哪条链上，并写入chain列).
    :param dataset_path: S1932.csv的路径.
    :return: DataFrame.
    """

    pdbDataset_path = '../datasets/%s/pdb%s/' % (csv_path.split('/')[2], csv_path.split('/')[2])
    atom_dict = {'A': 'Ala', 'R': 'Arg', 'N': 'Asn', 'D': 'Asp', 'C': 'Cys',
                 'Q': 'Gln', 'E': 'Glu', 'G': 'Gly', 'H': 'His', 'I': 'Ile',
                 'L': 'Leu', 'K': 'Lys', 'M': 'Met', 'F': 'Phe', 'P': 'Pro',
                 'S': 'Ser', 'T': 'Thr', 'W': 'Trp', 'Y': 'Tyr', 'V': 'Val'}
    parser = PDBParser(PERMISSIVE=1)
    f = open(csv_path,'r')
    df1 = pd.read_csv(f)
    f.close()
    df_mutation = pd.DataFrame({'PDB':[],'WILD_TYPE':[],'MUTATION':[],'CHAIN':[],'POSITION':[],'MUTANT':[],'PH':[],
                                'TEMPERATURE':[],'DDG':[],'RSA':[]})
    df1.loc[:,'POSITION'] = df1.MUTATION.str[1:-1]

    pdb_list = list(df1.drop_duplicates('PDB', 'first', inplace=False).PDB)

    for pdbid in pdb_list:
        df_pdbid = df1.loc[df1.PDB == pdbid,:]
        structure = parser.get_structure(pdbid, pdbDataset_path+'%s.pdb'%pdbid)
        chains = structure[0].get_list()

        for i in range(len(df_pdbid)):
            row = df_pdbid.iloc[i,:]
            wild_type, position = row.WILD_TYPE, int(row.POSITION)
            for chain in chains:
                if chain.has_id(position) and chain[position].get_resname().lower() == atom_dict[wild_type].lower():
                    df_pdbid.iloc[i,3] = chain.get_id()
                    break
        df_mutation = pd.concat([df_mutation,df_pdbid],axis=0,ignore_index=True)

    #df_mutation.pop('MUTATION')
    key = np.arange(len(df_mutation))
    df_mutation = pd.DataFrame({'key':key,'PDB':df_mutation.PDB,'WILD_TYPE':df_mutation.WILD_TYPE,'CHAIN':df_mutation.CHAIN,
                                'POSITION':df_mutation.POSITION,'MUTANT':df_mutation.MUTANT,'PH':df_mutation.PH,
                                'TEMPERATURE':df_mutation.TEMPERATURE,'DDG':df_mutation.DDG,'RSA':df_mutation.RSA})
    return df_mutation



if __name__ == '__main__':

    dataset_name, func = sys.argv[1:]
    dataset_path = '../datasets/%s/'%(dataset_name)
    csv_path = dataset_path+'%s.csv'%(dataset_name)

    if int(func) == 0:
        df_mutation = split_csv(csv_path)
    elif int(func) == 1:
        df_mutation = process_csv(csv_path)

    df_mutation.to_csv(dataset_path+'%s_new.csv'%(dataset_name), index=False)
    # print(df_mutation)