#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import sys, time
import warnings
from Bio import BiopythonWarning
from Bio.PDB.PDBParser import PDBParser
warnings.simplefilter('ignore', BiopythonWarning)

dataset_name = 'S2648'
path_csv_mutation = '../datasets/%s/%s_new_test.csv'%(dataset_name, dataset_name)

datadir = '/public/home/sry/mCNN/datasets/%s/pdb%s'%(dataset_name, dataset_name) #pdbS2648
mdlid=0
aa_dict = {'Ala': 'A', 'Arg': 'R', 'Asn': 'N', 'Asp': 'D', 'Cys': 'C',
           'Gln': 'Q', 'Glu': 'E', 'Gly': 'G', 'His': 'H', 'Ile': 'I',
           'Leu': 'L', 'Lys': 'K', 'Met': 'M', 'Phe': 'F', 'Pro': 'P',
           'Ser': 'S', 'Thr': 'T', 'Trp': 'W', 'Tyr': 'Y', 'Val': 'V'} # from wiki
def pdb2seq(seqname, filename, mdlid, chainid, wtflag, position='0', mtaa = '0'):
    lst = []
    parser = PDBParser(PERMISSIVE=1)
    structure = parser.get_structure(filename,'../datasets/S2648/pdbS2648/' + filename + '.pdb')
    model = structure[int(mdlid)]
    chain = model[chainid]
    if wtflag == 'WT':
        for residue in chain:
            res_id = residue.get_id()
            if res_id[0] == ' ':
                long_name = chain[res_id].get_resname()
                short_name = aa_dict[long_name[0]+long_name[1].lower()+long_name[2].lower()]
                lst.append(short_name)
            # print(lst)
    if wtflag=='MT':
        if position.isdigit():
            mutid = (' ',int(position),' ')
        else:
            mutid = (' ',int(position[:-1]),position[-1])
        for residue in chain:
            res_id = residue.get_id()
            if res_id[0] == ' ' and res_id != mutid:
                long_name = chain[res_id].get_resname()
                short_name = aa_dict[long_name[0]+long_name[1].lower()+long_name[2].lower()]
                lst.append(short_name)
            elif res_id == mutid:
                lst.append(mtaa)

    # print(len(lst))
    # print(len(set(lst)))
    fasta_name = '%s.fasta'%seqname
    g = open(fasta_name, 'w+')
    g.writelines('>%s.fasta|user:sry|date:%s|mdl:%s|chain:%s|pos:%s|mt_res:%s\n'
                 %(seqname[-12:], time.strftime("%a %b %d %H:%M:%S %Y",time.localtime()),
                   mdlid, chainid, position, mtaa))
    # print(len(lst))
    # print(lst)
    g.writelines(''.join(aa for aa in lst))
    g.writelines('\n')
    g.close()

def main():
    f = open(path_csv_mutation, 'r')
    df_mutation = pd.read_csv(f)
    f.close()
    df_wt = df_mutation.loc[:,['PDB', 'CHAIN']]
    df_wt.drop_duplicates(keep='first', inplace=True)
    wt_num = len(df_wt)
    mut_num = len(df_mutation)
    for i in range(wt_num):
        wt_index = df_wt.iloc[i, :].values
        filename, chainid = wt_index[0][0:4] , wt_index[1]
        seqname = '../datasets/%s/seq%s/WT_%s_%04d'%(dataset_name,dataset_name,filename,i+1)
        wtflag = 'WT'
        position = '0'
        mtaa = '0'
        pdb2seq(seqname, filename, mdlid, chainid, wtflag, position, mtaa)
    # =========================================================
    for i in range(mut_num):
        wtflag = 'MT'
        mt_index = df_mutation.loc[i, ['PDB','CHAIN','POSITION', 'MUTANT']].values
        filename, chainid, position, mtaa = mt_index[0][0:4], mt_index[1], mt_index[2], mt_index[3]
        seqname = '../datasets/%s/seq%s/MT_%s_%04d'%(dataset_name,dataset_name,filename,i+1)
        pdb2seq(seqname, filename, mdlid, chainid, wtflag, position, mtaa)

if __name__ == '__main__':
    main()