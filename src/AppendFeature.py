#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
** Append features to each atom in csv file, which obtained by CalNeighbor.py **
** 10/10/2019.
** sry.
The csv column index are: [chain,res,het,posid,inode,full_name,dist,x,y,z,occupancy,b_factor]
'''
import os, argparse
import numpy as np
import pandas as pd
aa_321dict = {'Ala': 'A', 'Arg': 'R', 'Asn': 'N', 'Asp': 'D', 'Cys': 'C',
              'Gln': 'Q', 'Glu': 'E', 'Gly': 'G', 'His': 'H', 'Ile': 'I',
              'Leu': 'L', 'Lys': 'K', 'Met': 'M', 'Phe': 'F', 'Pro': 'P',
              'Ser': 'S', 'Thr': 'T', 'Trp': 'W', 'Tyr': 'Y', 'Val': 'V'}  # from wiki
aa_atom_dict = {'A': {'C': 3, 'H': 7, 'O': 2, 'N': 1},
                'R': {'C': 6, 'H': 14, 'O': 2, 'N': 4},
                'N': {'C': 4, 'H': 8, 'O': 3, 'N': 2},
                'D': {'C': 4, 'H': 7, 'O': 4, 'N': 1},
                'C': {'C': 3, 'H': 7, 'O': 2, 'N': 1, 'S': 1},
                'Q': {'C': 5, 'H': 10, 'O': 3, 'N': 2},
                'E': {'C': 5, 'H': 9, 'O': 4, 'N': 1},
                'G': {'C': 2, 'H': 5, 'O': 2, 'N': 1},
                'H': {'C': 6, 'H': 9, 'O': 2, 'N': 3},
                'I': {'C': 6, 'H': 13, 'O': 2, 'N': 1},
                'L': {'C': 6, 'H': 13, 'O': 2, 'N': 1},
                'K': {'C': 6, 'H': 14, 'O': 2, 'N': 2},
                'M': {'C': 5, 'H': 11, 'O': 2, 'N': 1, 'S': 1},
                'F': {'C': 9, 'H': 11, 'O': 2, 'N': 1},
                'P': {'C': 5, 'H': 9, 'O': 2, 'N': 1},
                'S': {'C': 3, 'H': 7, 'O': 3, 'N': 1},
                'T': {'C': 4, 'H': 9, 'O': 3, 'N': 1},
                'W': {'C': 11, 'H': 12, 'O': 2, 'N': 2},
                'Y': {'C': 9, 'H': 11, 'O': 3, 'N': 1},
                'V': {'C': 5, 'H': 11, 'O': 2, 'N': 1}}
aa_vec_dict = {}  # aa_vec_dict = {'aa_name':[vec_of_atom_number by class], ...}
## Calc aa_vec_dict.
for aa_name in aa_atom_dict.keys():
    aa_vec = list(aa_atom_dict[aa_name].values())
    if len(aa_vec) == 4:
        aa_vec.append(0)
    aa_vec_dict[aa_name] = aa_vec

def calEntropy(blastdir,position):
    filedir = '%s/msa.cnt_frq.npz'%blastdir
    data = np.load(filedir,allow_pickle=True)
    frq = data['frq']
    position_index = frq[1:,1]
    index = np.argwhere(position_index==str(position))+1
    rv = frq[index, 3:].astype(float).reshape(-1)
    rv_pop = list(filter(lambda x:x!=0,rv))
    entropy = sum(list(map(lambda x:-x * np.log2(x), rv_pop)))
    return rv, entropy

def append_feature(df,feature,filename,pH,T,ddg,rsa=0):
    len_df = len(df)
    dataset_name, pdb, wtaa, mtchain, mtposition, mtaa, serial = filename.split('_')
    if feature == 'rsa':
        temp_df = pd.DataFrame(np.ones((len_df,1)) * rsa)
        df['rsa'] = temp_df
        return df
    if feature == 'thermo':
        temp_df_ph = pd.DataFrame(np.ones((len_df, 1)) * pH)
        temp_df_t  = pd.DataFrame(np.ones((len_df, 1)) * T)
        df['ph'] = temp_df_ph
        df['temperature'] = temp_df_t
        return df
    if feature == 'onehot':
        # columnlst = list(df.columns) # chain,res,het,posid,inode,full_name,dist,x,y,z,occupancy,b_factor
        temp_df = pd.DataFrame(np.zeros((len_df,1)))
        df['C'] = temp_df
        df['O'] = temp_df
        df['N'] = temp_df
        df['Other'] = temp_df
        df.loc[df.atom_name == 'C', 'C'] = 1
        df.loc[df.atom_name == 'O', 'O'] = 1
        df.loc[df.atom_name == 'N', 'N'] = 1
        df.loc[(df.atom_name != 'C') & (df.full_name != 'O') & (df.full_name != 'N'), 'Other'] = 1
        return df
    if feature == 'deltar':
        delta_r = np.array(aa_vec_dict[mtaa]) - np.array(aa_vec_dict[wtaa])
        temp_df = pd.DataFrame(np.ones((len_df, 1)))
        df['dC'] = temp_df * delta_r[0]
        df['dH'] = temp_df * delta_r[1]
        df['dO'] = temp_df * delta_r[2]
        df['dN'] = temp_df * delta_r[3]
        df['dOther'] = temp_df * delta_r[4]
        return df
    if feature == 'msa':
        entropylst = []
        WTmsalst = []
        MTmsalst = []
        for i in range(len_df):
            atom_chain, atom_res, atom_het, atom_posid, atom_inode, atom_full_name,atom_name = df.iloc[i,:].values[0:7]
            atom_position = str(atom_het)+str(atom_posid)+str(atom_inode)
            atom_position = atom_position.strip()
            pattern = 'psiblast_WT_%s_%s'%(pdb,atom_chain)
            wtdirlst = os.listdir('/public/home/sry/mCNN/msa/psiblast%s'%dataset_name)
            indexlst = [i for i,x in enumerate(wtdirlst) if x.find(pattern)!=-1]
            wt_blastdir = '/public/home/sry/mCNN/msa/psiblast%s/%s'%(dataset_name,wtdirlst[indexlst[0]])
            rvWT, entWT = calEntropy(wt_blastdir,atom_position)

            if atom_chain == mtchain:
                mt_blastdir = '/public/home/sry/mCNN/msa/psiblast%s/psiblast_MT_%s_%s_%s_%s_%s_%s' % (
                    dataset_name, pdb, wtaa, mtchain, mtposition, mtaa, serial)
                rvMT, entMT = calEntropy(mt_blastdir, atom_position)
            else:
                rvMT, entMT = calEntropy(wt_blastdir,atom_position)
            ent = entMT - entWT
            entropylst.append(ent)
            WTmsalst.append(rvWT)
            MTmsalst.append(rvMT)
        temp_df = pd.DataFrame(np.array(entropylst).reshape(len_df,1))
        df['dEntropy'] = temp_df
        cols = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', '-']
        MTcols = ['MT_'+aa for aa in cols]
        WTcols = ['WT_'+aa for aa in cols]
        WTmsadf = pd.DataFrame(np.array(WTmsalst).reshape(len_df, 21),columns=WTcols)
        MTmsadf = pd.DataFrame(np.array(MTmsalst).reshape(len_df, 21),columns=MTcols)

        df = pd.concat([df,WTmsadf,MTmsadf],axis=1)
        return df

    if feature == 'ddg':
        temp_df = pd.DataFrame(np.ones((len_df, 1)) * ddg)
        df['ddg'] = temp_df
        return df

def save_csv(DF, FILENAME, OUTDIR='.'):
    DF.to_csv('%s/%s.csv'%(OUTDIR,FILENAME), index=False)
if __name__ == '__main__':
    ## input parameters in shell.
    parser = argparse.ArgumentParser()
    parser.description = '* Append features for each atom in csv file which obtained by CalNeighbor.py'
    parser.add_argument('csvdir',            type=str,   help='The input directory of a csv file to which going to append')
    parser.add_argument('filename',          type=str,   help='The output file name, consist of The index of each mutation')
    parser.add_argument('-o', '--outdir',    type=str,   default='.', help='The output directory, default="."')
    parser.add_argument('-f', '--feature',   nargs='*',  type=str,    choices=['rsa','thermo','onehot','deltar','msa','ddg'], default='', help='The feature to append, default=""')
    parser.add_argument('-r', '--rsa',       type=str,   help='The RSA value to append')
    parser.add_argument('-t', '--thermo',    nargs=2,    type=float,  help='The pH and Temperature value to append')
    parser.add_argument('-d', '--ddg',       type=str,   help='The DDG value to append')

    args     = parser.parse_args()
    CSVDIR   = args.csvdir
    FILENAME = args.filename
    if args.outdir:
        OUTDIR = args.outdir
        if not os.path.exists(OUTDIR):
            os.mkdir(OUTDIR)
    if args.rsa:
        RSA = float(args.rsa)
    if args.thermo:
        THERMO = args.thermo
    if args.ddg:
        DDG = float(args.ddg)
        print('ddg get')

    if not args.feature:
        print('Nothing to do!')
    else:
        f = open(CSVDIR, 'r')
        df = pd.read_csv(f)
        f.close()
        for FEATURE in args.feature:
            if args.rsa:
                df = append_feature(df,FEATURE,FILENAME,THERMO[0],THERMO[1],DDG,RSA)#df,feature,filename,pH,T,ddg,rsa=0
            else:
                df = append_feature(df, FEATURE, FILENAME, THERMO[0], THERMO[1], DDG)

        save_csv(df, FILENAME, OUTDIR=OUTDIR)