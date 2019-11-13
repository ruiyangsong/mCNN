#!/usr/bin/env python
# -*- coding: utf-8 -*-

'file name format: WT_pdb_chain_serial | MT_pdb_wtaa_chain_position_mtaa_serial'

import pandas as pd
import sys, time
import warnings
from Bio import BiopythonWarning
from Bio.PDB.PDBParser import PDBParser
warnings.simplefilter('ignore', BiopythonWarning)

# dataset_name = 'S2648'
dataset_name = sys.argv[1]
path_csv_mutation = '../datasets/%s/%s_new.csv'%(dataset_name, dataset_name)

datadir = '/public/home/sry/mCNN/datasets/%s/pdb%s'%(dataset_name, dataset_name) #pdbS2648
mdlid=0
aa_dict = {'Ala': 'A', 'Arg': 'R', 'Asn': 'N', 'Asp': 'D', 'Cys': 'C',
           'Gln': 'Q', 'Glu': 'E', 'Gly': 'G', 'His': 'H', 'Ile': 'I',
           'Leu': 'L', 'Lys': 'K', 'Met': 'M', 'Phe': 'F', 'Pro': 'P',
           'Ser': 'S', 'Thr': 'T', 'Trp': 'W', 'Tyr': 'Y', 'Val': 'V'} # from wiki
def pdb2seq(seqname, filename, mdlid, chainid, wtflag, position='0', wtaa = '0',mtaa = '0'):
    lst = []
    indexlst = []
    parser = PDBParser(PERMISSIVE=1)
    structure = parser.get_structure(filename,'../datasets/%s/pdb%s/%s.pdb'%(dataset_name,dataset_name,filename))
    model = structure[int(mdlid)]
    chain = model[chainid]
    if wtflag == 'WT':
        for residue in chain:
            res_id = residue.get_id()
            if res_id[0] == ' ':
                if res_id[2] == ' ':
                    index = str(res_id[1])
                else:
                    index = str(res_id[1]) + str(res_id[2])

                long_name = chain[res_id].get_resname()
                # print(filename,chain.get_id(),res_id,long_name) # nucleic acid occurs
                try:
                    short_name = aa_dict[long_name[0]+long_name[1].lower()+long_name[2].lower()]
                    indexlst.append(index)
                    lst.append(short_name)
                except:
                    return 0
                    # print('nucleic acid occurs in WT,filename:%s|chain_id:%s|res_id:%s|long_name:%s'%(filename,chain.get_id(),res_id,long_name))

            # print(lst)
    if wtflag=='MT':
        if dataset_name == 'S2648':
            if position.isdigit():
                mutid = (' ',int(position),' ')
            else:
                mutid = (' ',int(position[:-1]),position[-1])
        elif dataset_name == 'S1925':
            mutid = (' ', int(position), ' ')
        for residue in chain:
            res_id = residue.get_id()
            if res_id[0] == ' ' and res_id != mutid:
                if res_id[2] == ' ':
                    index = str(res_id[1])
                else:
                    index = str(res_id[1]) + str(res_id[2])

                long_name = chain[res_id].get_resname()
                try:
                    short_name = aa_dict[long_name[0]+long_name[1].lower()+long_name[2].lower()]
                    indexlst.append(index)
                    lst.append(short_name)
                except:
                    print('nucleic acid occurs in MT! something may be wrong!,filename:%s|chain_id:%s|res_id:%s|long_name:%s'
                          %(filename,chain.get_id(),res_id,long_name))
            elif res_id == mutid:
                if res_id[2] == ' ':
                    index = str(res_id[1])
                else:
                    index = str(res_id[1]) + str(res_id[2])
                indexlst.append(index)
                lst.append(mtaa)

    # print(len(lst))
    # print(len(set(lst)))
    fasta_name = '%s.fasta'%seqname
    g = open(fasta_name, 'w+')
    g.writelines('>%s,%s.fasta|user:sry|date:%s|mdl:%s|chain:%s|pos:%s|wt_res:%s|mt_res:%s\n'
                 %(','.join(indexlst),seqname.split('/')[-1], time.strftime("%a %b %d %H:%M:%S %Y",time.localtime()),
                   mdlid, chainid, position, wtaa, mtaa))
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
        wtflag = 'WT'
        position = '0'
        mtaa = '0'
        wtaa = '0'
        wt_index = df_wt.iloc[i, :].values
        filename = wt_index[0][0:4]
        # chainid =  wt_index[1]
        parser_tmp = PDBParser(PERMISSIVE=1)
        structure_tmp = parser_tmp.get_structure(filename,'../datasets/%s/pdb%s/%s.pdb' % (dataset_name, dataset_name, filename))
        model_tmp = structure_tmp[int(mdlid)]
        for chain_tmp in model_tmp:
            chainid = chain_tmp.get_id()
            seqname = '../datasets/%s/seq%s/WT_%s_%s_%04d'%(dataset_name,dataset_name,filename,chainid,i+1)
            pdb2seq(seqname, filename, mdlid, chainid, wtflag, position,wtaa, mtaa)
    # =========================================================
    for i in range(mut_num):
        wtflag = 'MT'
        mt_index = df_mutation.loc[i, ['PDB','WILD_TYPE','CHAIN','POSITION', 'MUTANT']].values
        filename, wtaa, chainid, position, mtaa = mt_index[0][0:4], mt_index[1], mt_index[2], mt_index[3], mt_index[4]
        seqname = '../datasets/%s/seq%s/MT_%s_%s_%s_%s_%s_%04d'\
                  %(dataset_name,dataset_name,filename,wtaa,chainid,position,mtaa,i+1)
        pdb2seq(seqname, filename, mdlid, chainid, wtflag, position, wtaa, mtaa)

if __name__ == '__main__':
    main()
