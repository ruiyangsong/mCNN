#!/usr/bin/env python
import os,warnings
from functools import reduce
import numpy as np
import pandas as pd
from Bio import BiopythonWarning
from Bio.PDB.PDBParser import PDBParser

def main():
    pdbdir = '/public/home/sry/mCNN/dataset/TR/pdb_chain'
    for filename in os.listdir(pdbdir):
        pdbpth = '/public/home/sry/mCNN/dataset/TR/pdb_chain/%s'%filename
        outdir = '/public/home/sry/mCNN/dataset/TR/map_pos'
        pos_map_filename = filename[:4]
        mapping(pdbpth,outdir,pos_map_filename)

def mapping(pdbpth,outdir,pos_map_filename):
    """
    Creating Position-Mapping file (csv) of a pdb file to get consecutive numbering.
    Columns in a Position-Mapping file are: ['CHAIN',POSITION_OLD','POSITION_NEW']
    :param pdbpth: str, path of pdb file to mapping position
    :param outdir: str, output directory of Position-Mapping file
    :return: Position-Mapping dataframe
    """
    if not os.path.exists(pdbpth):
        raise FileNotFoundError('File %s Not Found'%pdbpth)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    chain_id_lst = [] #[all_chainID_in_this_pdbMDL]
    pos_old_dict = {} #{'chain_id': [stripped res_id_old]}
    pos_new_dict = {}

    warnings.simplefilter('ignore', BiopythonWarning)
    pdbid = pdbpth.split('/')[-1].split('.')[0]
    parser = PDBParser(PERMISSIVE=1)
    structure = parser.get_structure(pdbid, pdbpth)
    model = structure[0]
    for chain in model:
        chain_id = chain.get_id() #chain_id[A, B, C, ...]
        res_id_lst = list(map(lambda tup: (tup[0] + str(tup[1]) + tup[2]).strip(), [res.get_id() for res in chain])) #res_id: (' ', 154, ' ')
        if len(res_id_lst) > 0:
            chain_id_lst.append(chain_id)
            pos_old_dict[chain_id] = res_id_lst
        else:
            print('[WARNING] pdbpth: %s, chain: %s do not contains any STANDARD residue.' % (pdbpth, chain_id))
    try:
        assert len(chain_id_lst) > 0
    except:
        print('[ERROR] pdbpth: %s do not contains any STANDARD residue, some badthings may happen',exc_info=True)
        exit(1)

    chain_len_lst = [len(pos_old_dict[chain_id]) for chain_id in chain_id_lst] #[res_num_of_each_chain]
    df_chain_arr = []
    df_pos_old_arr = []
    df_pos_new_arr = []
    for i in range(len(chain_id_lst)):
        chain_id = chain_id_lst[i]
        chain_len = chain_len_lst[i]
        ## residues in the first chain are numbering from "1" to "len(first_chain)"
        ## residues in the following chain are numbering from the "former_serial_number+1" to "former_serial_number+1+len(this_chain)"
        if i == 0:
            pos_new_dict[chain_id] = list(np.arange(1, chain_len + 1))
        if i > 0:
            cum_sum = reduce(lambda x, y: x + y, chain_len_lst[:i])
            pos_new_dict[chain_id] = list(np.arange(1, chain_len + 1) + cum_sum)
        ## dict to dataframe
        df_chain_arr = df_chain_arr + [chain_id for _ in range(chain_len)]
        df_pos_old_arr = df_pos_old_arr + pos_old_dict[chain_id]
        df_pos_new_arr = df_pos_new_arr + pos_new_dict[chain_id]
    df_mapping = pd.DataFrame({'CHAIN': df_chain_arr, 'POSITION_OLD': df_pos_old_arr, 'POSITION_NEW': df_pos_new_arr})
    df_mapping.to_csv('%s/%s_mapping.csv' % (outdir, pos_map_filename), index=False)

    df_mapping['CHAIN'] = df_mapping['CHAIN'].astype(str)
    df_mapping['POSITION_OLD'] = df_mapping['POSITION_OLD'].astype(str)
    df_mapping['POSITION_NEW'] = df_mapping['POSITION_NEW'].astype(str)

    return df_mapping
    
if __name__ == '__main__':
    main()