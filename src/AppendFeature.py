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

stride_secondary = {'H':'H','G':'H','I':'H','E':'E','B':'C','T':'C','C':'C'} #https://ssbio.readthedocs.io/en/latest/instructions/stride.html

ASA_dict = {'A': 110.2, 'C': 140.4, 'D': 144.1, 'E': 174.7, 'F': 200.7,
            'G': 78.7,  'H': 181.9, 'I': 185.0, 'K': 205.7, 'L': 183.1,
            'M': 200.1, 'N': 146.4, 'P': 141.9, 'Q': 178.6, 'R': 229.0,
            'S': 117.2, 'T': 138.7, 'V': 153.7, 'W': 240.5, 'Y': 213.7}

# secondary_lst = ['G', 'H', 'b', 'C', 'T', 'B', 'E']
aa_atom_mass = {'C': 12.0107, 'H': 1.0079, 'O': 15.9994, 'N': 14.0067, 'S': 32.065} # from https://www.lenntech.com/periodic/mass/atomic-mass.htm

aa_321dict = {'Ala': 'A', 'Arg': 'R', 'Asn': 'N', 'Asp': 'D', 'Cys': 'C',
              'Gln': 'Q', 'Glu': 'E', 'Gly': 'G', 'His': 'H', 'Ile': 'I',
              'Leu': 'L', 'Lys': 'K', 'Met': 'M', 'Phe': 'F', 'Pro': 'P',
              'Ser': 'S', 'Thr': 'T', 'Trp': 'W', 'Tyr': 'Y', 'Val': 'V'}  # from wiki
# aa_3lst = ['Ala', 'Arg', 'Asn', 'Asp', 'Cys', 'Gln', 'Glu', 'Gly', 'His', 'Ile', 'Leu', 'Lys', 'Met', 'Phe', 'Pro', 'Ser', 'Thr', 'Trp', 'Tyr', 'Val']
aa_atom_dict = {'A': {'C': 3,  'H': 7,  'O': 2, 'N': 1},
                'R': {'C': 6,  'H': 14, 'O': 2, 'N': 4},
                'N': {'C': 4,  'H': 8,  'O': 3, 'N': 2},
                'D': {'C': 4,  'H': 7,  'O': 4, 'N': 1},
                'C': {'C': 3,  'H': 7,  'O': 2, 'N': 1, 'S': 1},
                'Q': {'C': 5,  'H': 10, 'O': 3, 'N': 2},
                'E': {'C': 5,  'H': 9,  'O': 4, 'N': 1},
                'G': {'C': 2,  'H': 5,  'O': 2, 'N': 1},
                'H': {'C': 6,  'H': 9,  'O': 2, 'N': 3},
                'I': {'C': 6,  'H': 13, 'O': 2, 'N': 1},
                'L': {'C': 6,  'H': 13, 'O': 2, 'N': 1},
                'K': {'C': 6,  'H': 14, 'O': 2, 'N': 2},
                'M': {'C': 5,  'H': 11, 'O': 2, 'N': 1, 'S': 1},
                'F': {'C': 9,  'H': 11, 'O': 2, 'N': 1},
                'P': {'C': 5,  'H': 9,  'O': 2, 'N': 1},
                'S': {'C': 3,  'H': 7,  'O': 3, 'N': 1},
                'T': {'C': 4,  'H': 9,  'O': 3, 'N': 1},
                'W': {'C': 11, 'H': 12, 'O': 2, 'N': 2},
                'Y': {'C': 9,  'H': 11, 'O': 3, 'N': 1},
                'V': {'C': 5,  'H': 11, 'O': 2, 'N': 1}}

aa_atom_pharm_dict = {'ALA':
                          {'N'   :[0,0,0,1,0,1,0,0], 'C'   :[0,0,0,1,0,0,0,0], 'O'   :[0,0,0,1,1,0,0,0], 'CA'  :[0,0,0,1,0,0,0,0],
                           'CB'  :[1,0,0,1,0,0,0,0]},
                      'ARG':
                          {'N'   :[0,0,0,1,0,1,0,0], 'C'   :[0,0,0,1,0,0,0,0], 'O'   :[0,0,0,1,1,0,0,0], 'CA'  :[0,0,0,1,0,0,0,0],
                           'CB'  :[1,0,0,1,0,0,0,0], 'CG'  :[1,0,0,1,0,0,0,0], 'CD'  :[0,0,0,1,0,0,0,0], 'NE'  :[0,0,0,1,0,1,0,0],
                           'CZ'  :[0,0,0,1,0,0,0,0], 'NH1' :[0,1,0,0,0,1,0,0], 'NH2' :[0,1,0,0,0,1,0,0]},
                      'ASN':
                          {'N'   :[0,0,0,1,0,1,0,0], 'C'   :[0,0,0,1,0,0,0,0], 'O'   :[0,0,0,1,1,0,0,0], 'CA'  :[0,0,0,1,0,0,0,0],
                           'CB'  :[1,0,0,1,0,0,0,0], 'CG'  :[0,0,0,1,0,0,0,0], 'OD1' :[0,0,0,1,1,0,0,0], 'ND2' :[0,0,0,1,0,1,0,0]},
                      'ASP':
                          {'N'   :[0,0,0,1,0,1,0,0], 'C'   :[0,0,0,1,0,0,0,0], 'O'   :[0,0,0,1,1,0,0,0], 'CA'  :[0,0,0,1,0,0,0,0],
                           'CB'  :[1,0,0,1,0,0,0,0], 'CG'  :[0,0,0,1,0,0,0,0], 'OD1' :[0,0,1,0,1,0,0,0], 'OD2' :[0,0,1,0,1,0,0,0]},
                      'CYS':
                          {'N'   :[0,0,0,1,0,1,0,0], 'H'   :[0,0,0,1,0,0,0,0], 'C'   :[0,0,0,1,0,0,0,0], 'O'   :[0,0,0,1,1,0,0,0],
                           'CA'  :[0,0,0,1,0,0,0,0], 'CB'  :[1,0,0,1,0,0,0,0], 'SG'  :[1,0,0,1,0,0,0,1]},
                      'GLN':
                          {'N'   :[0,0,0,1,0,1,0,0], 'C'   :[0,0,0,1,0,0,0,0], 'O'   :[0,0,0,1,1,0,0,0], 'CA'  :[0,0,0,1,0,0,0,0],
                           'CB'  :[1,0,0,1,0,0,0,0], 'CG'  :[1,0,0,1,0,0,0,0], 'CD'  :[0,0,0,1,0,0,0,0], 'OE1' :[0,0,0,1,1,0,0,0],
                           'NE2' :[0,0,0,1,0,1,0,0]},
                      'GLU':
                          {'N'   :[0,0,0,1,0,1,0,0], 'C'   :[0,0,0,1,0,0,0,0], 'O'   :[0,0,0,1,1,0,0,0], 'CA'  :[0,0,0,1,0,0,0,0],
                           'CB'  :[1,0,0,1,0,0,0,0], 'CG'  :[1,0,0,1,0,0,0,0], 'CD'  :[0,0,0,1,0,0,0,0], 'OE1' :[0,0,1,0,1,0,0,0],
                           'OE2' :[0,0,1,0,1,0,0,0]},
                      'GLY':
                          {'N'   :[0,0,0,1,0,1,0,0], 'C'   :[0,0,0,1,0,0,0,0], 'O'   :[0,0,0,1,1,0,0,0], 'CA'  :[0,0,0,1,0,0,0,0]},
                      'HIS':
                          {'N'   :[0,0,0,1,0,1,0,0], 'C'   :[0,0,0,1,0,0,0,0], 'O'   :[0,0,0,1,1,0,0,0], 'CA'  :[0,0,0,1,0,0,0,0],
                           'CB'  :[1,0,0,1,0,0,0,0], 'CG'  :[0,0,0,1,0,0,1,0], 'ND1' :[0,1,0,0,1,1,1,0], 'CD2' :[0,0,0,1,0,0,1,0],
                           'CE1' :[0,0,0,1,0,0,1,0], 'NE2' :[0,1,0,0,1,1,1,0]},
                      'ILE':
                          {'N'   :[0,0,0,1,0,1,0,0], 'C'   :[0,0,0,1,0,0,0,0], 'O'   :[0,0,0,1,1,0,0,0], 'CA'  :[0,0,0,1,0,0,0,0],
                           'CB'  :[1,0,0,1,0,0,0,0], 'CG2' :[1,0,0,1,0,0,0,0], 'CG1' :[1,0,0,1,0,0,0,0], 'CD1' :[1,0,0,1,0,0,0,0]},
                      'LEU':
                          {'N'   :[0,0,0,1,0,1,0,0], 'C'   :[0,0,0,1,0,0,0,0], 'O'   :[0,0,0,1,1,0,0,0], 'CA'  :[0,0,0,1,0,0,0,0],
                           'CB'  :[1,0,0,1,0,0,0,0], 'CG'  :[1,0,0,1,0,0,0,0], 'CD1' :[1,0,0,1,0,0,0,0], 'CD2' :[1,0,0,1,0,0,0,0]},
                      'LYS':
                          {'N'   :[0,0,0,1,0,1,0,0], 'C'   :[0,0,0,1,0,0,0,0], 'O'   :[0,0,0,1,1,0,0,0], 'CA'  :[0,0,0,1,0,0,0,0],
                           'CB'  :[1,0,0,1,0,0,0,0], 'CG'  :[1,0,0,1,0,0,0,0], 'CD'  :[1,0,0,1,0,0,0,0], 'CE'  :[0,0,0,1,0,0,0,0],
                           'NZ'  :[0,1,0,0,0,1,0,0]},
                      'MET':
                          {'N'   :[0,0,0,1,0,1,0,0], 'C'   :[0,0,0,1,0,0,0,0], 'O'   :[0,0,0,1,1,0,0,0], 'CA'  :[0,0,0,1,0,0,0,0],
                           'CB'  :[1,0,0,1,0,0,0,0], 'CG'  :[1,0,0,1,0,0,0,0], 'SD'  :[1,0,0,1,0,0,0,1], 'CE'  :[1,0,0,1,0,0,0,0]},
                      'PHE':
                          {'N'   :[0,0,0,1,0,1,0,0], 'C'   :[0,0,0,1,0,0,0,0], 'O'   :[0,0,0,1,1,0,0,0], 'CA'  :[0,0,0,1,0,0,0,0],
                           'CB'  :[1,0,0,1,0,0,0,0], 'CG'  :[1,0,0,1,0,0,1,0], 'CD1' :[1,0,0,1,0,0,1,0], 'CD2' :[1,0,0,1,0,0,1,0],
                           'CE1' :[1,0,0,1,0,0,1,0], 'CE2' :[1,0,0,1,0,0,1,0], 'CZ'  :[1,0,0,1,0,0,1,0]},
                      'PRO':
                          {'N'   :[0,0,0,1,0,0,0,0], 'C'   :[0,0,0,1,0,0,0,0], 'O'   :[0,0,0,1,1,0,0,0], 'CD'  :[0,0,0,1,0,0,0,0],
                           'CA'  :[0,0,0,1,0,0,0,0], 'CB'  :[1,0,0,1,0,0,0,0], 'CG'  :[1,0,0,1,0,0,0,0]},
                      'SER':
                          {'N'   :[0,0,0,1,0,1,0,0], 'C'   :[0,0,0,1,0,0,0,0], 'O'   :[0,0,0,1,1,0,0,0], 'CA'  :[0,0,0,1,0,0,0,0],
                           'CB'  :[0,0,0,1,0,0,0,0], 'OG'  :[0,0,0,1,1,1,0,0]},
                      'THR':
                          {'N'   :[0,0,0,1,0,1,0,0], 'C'   :[0,0,0,1,0,0,0,0], 'O'   :[0,0,0,1,1,0,0,0], 'CA'  :[0,0,0,1,0,0,0,0],
                           'CB'  :[0,0,0,1,0,0,0,0], 'OG1' :[0,0,0,1,1,1,0,0], 'CG2' :[1,0,0,1,0,0,0,0]},
                      'TRP':
                          {'N'   :[0,0,0,1,0,1,0,0], 'C'   :[0,0,0,1,0,0,0,0], 'O'   :[0,0,0,1,1,0,0,0], 'CA'  :[0,0,0,1,0,0,0,0],
                           'CB'  :[1,0,0,1,0,0,0,0], 'CG'  :[1,0,0,1,0,0,1,0], 'CD2' :[1,0,0,1,0,0,1,0], 'CE2' :[0,0,0,1,0,0,1,0],
                           'CE3' :[1,0,0,1,0,0,1,0], 'CD1' :[0,0,0,1,0,0,1,0], 'NE1' :[0,0,0,1,0,1,1,0], 'CZ2' :[1,0,0,1,0,0,1,0],
                           'CZ3' :[1,0,0,1,0,0,1,0], 'CH2' :[1,0,0,1,0,0,1,0]},
                      'TYR':
                          {'N'   :[0,0,0,1,0,1,0,0], 'C'   :[0,0,0,1,0,0,0,0], 'O'   :[0,0,0,1,1,0,0,0], 'CA'  :[0,0,0,1,0,0,0,0],
                           'CB'  :[1,0,0,1,0,0,0,0], 'CG'  :[1,0,0,1,0,0,1,0], 'CD1' :[1,0,0,1,0,0,1,0], 'CD2' :[1,0,0,1,0,0,1,0],
                           'CE1' :[1,0,0,1,0,0,1,0], 'CE2' :[1,0,0,1,0,0,1,0], 'CZ'  :[0,0,0,1,0,0,1,0], 'OH'  :[0,0,0,1,1,1,0,0]},
                      'VAL':
                          {'N'   :[0,0,0,1,0,1,0,0], 'C'   :[0,0,0,1,0,0,0,0], 'O'   :[0,0,0,1,1,0,0,0], 'CA'  :[0,0,0,1,0,0,0,0],
                           'CB'  :[1,0,0,1,0,0,0,0], 'CG1' :[1,0,0,1,0,0,0,0], 'CG2' :[1,0,0,1,0,0,0,0]}
                      }

aa_atom_hp_dict = {'ALA':
                       {'C':[0,1], 'CA':[0,1], 'CB':[1,0]},
                   'ARG':
                       {'C':[0,1], 'CA':[0,1], 'CB':[1,0], 'CG':[1,0], 'CD':[0,1], 'CZ':[0,1]},
                   'ASN':
                       {'C':[0,1], 'CA':[0,1], 'CB':[1,0], 'CG':[0,1]},
                   'ASP':
                       {'C':[0,1], 'CA':[0,1], 'CB':[1,0], 'CG':[0,1]},
                   'CYS':
                       {'C':[0,1], 'CA':[0,1], 'CB':[1,0], 'SG':[1,0]},
                   'GLN':
                       {'C':[0,1], 'CA':[0,1], 'CB':[1,0], 'CG':[1,0], 'CD':[0,1]},
                   'GLU':
                       {'C':[0,1], 'CA':[0,1], 'CB':[1,0], 'CG':[1,0], 'CD':[0,1]},
                   'GLY':
                       {'C':[0,1], 'CA':[0,1]},
                   'HIS':
                       {'C':[0,1], 'CA':[0,1], 'CB':[1,0], 'CG':[0,1], 'CD2' :[0,1], 'CE1':[0,1]},
                   'ILE':
                       {'C':[0,1], 'CA':[0,1], 'CB':[1,0], 'CG2':[1,0], 'CG1':[1,0], 'CD1':[1,0]},
                   'LEU':
                       {'C':[0,1], 'CA':[0,1], 'CB':[1,0], 'CG':[1,0], 'CD1' :[1,0], 'CD2':[1,0]},
                   'LYS':
                       {'C':[0,1], 'CA':[0,1], 'CB':[1,0], 'CG':[1,0], 'CD'  :[1,0], 'CE' :[0,1]},
                   'MET':
                       {'C':[0,1], 'CA':[0,1], 'CB':[1,0], 'CG':[1,0], 'SD'  :[1,0], 'CE' :[1,0]},
                   'PHE':
                       {'C':[0,1], 'CA':[0,1], 'CB':[1,0], 'CG':[1,0], 'CD1' :[1,0], 'CD2':[1,0], 'CE1':[1,0], 'CE2':[1,0], 'CZ':[1,0]},
                   'PRO':
                       {'N':[0,1], 'C' :[0,1], 'CD':[0,1], 'CA':[0,1], 'CB'  :[1,0], 'CG' :[1,0]},
                   'SER':
                       {'C':[0,1], 'CA':[0,1], 'CB':[0,1]},
                   'THR':
                       {'C':[0,1], 'CA':[0,1], 'CB':[0,1], 'CG2':[1,0]},
                   'TRP':
                       {'C':[0,1], 'CA':[0,1], 'CB':[1,0], 'CG' :[1,0], 'CD2':[1,0], 'CE2':[0,1], 'CE3':[1,0], 'CD1':[0,1], 'CZ2':[1,0], 'CZ3' :[1,0], 'CH2' :[1,0]},
                   'TYR':
                       {'C':[0,1], 'CA':[0,1], 'CB':[1,0], 'CG' :[1,0], 'CD1':[1,0], 'CD2':[1,0], 'CE1':[1,0], 'CE2':[1,0], 'CZ' :[0,1]},
                   'VAL':
                       {'C':[0,1], 'CA':[0,1], 'CB':[1,0], 'CG1':[1,0], 'CG2' :[1,0]}
                      }

aa_pharm_dict = {'A':[1,0,0,5,1,1,0,0],  'R':[2,2,0,9,1,4,0,0],  'N':[1,0,0,8,2,2,0,0],  'D':[1,0,2,6,3,1,0,0],
                 'C':[2,0,0,6,1,1,0,1],  'Q':[2,0,0,9,2,2,0,0],  'E':[2,0,2,7,3,1,0,0],  'G':[0,0,0,4,1,1,0,0],
                 'H':[1,2,0,8,3,3,5,0],  'I':[4,0,0,8,1,1,0,0],  'L':[4,0,0,8,1,1,0,0],  'K':[3,1,0,8,1,2,0,0],
                 'M':[4,0,0,8,1,1,0,1],  'F':[7,0,0,11,1,1,6,0], 'P':[2,0,0,7,1,0,0,0],  'S':[0,0,0,6,2,2,0,0],
                 'T':[1,0,0,7,2,2,0,0],  'W':[7,0,0,14,1,2,9,0], 'Y':[6,0,0,12,2,2,6,0], 'V':[3,0,0,7,1,1,0,0]}

aa_hp_dict = {'A':[1,2], 'R':[2,4], 'N':[1,3], 'D':[1,3], 'C':[2,2], 'Q':[2,3], 'E':[2,3], 'G':[0,2], 'H':[1,5], 'I':[4,2],
              'L':[4,2], 'K':[3,3], 'M':[4,2], 'F':[7,2], 'P':[2,4], 'S':[0,3], 'T':[1,3], 'W':[7,4], 'Y':[6,3], 'V':[3,2]}

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

def append_feature(df,feature,filename,pH,T,ddg,sadir):
    len_df = len(df)
    dataset_name, pdb, wtaa, mtchain, mtposition, mtaa, serial = filename.split('_')

    if feature == 'rsa':
        secondarylst = []
        salst  = []
        rsalst = []
        asalst = []
        philst = []
        psilst = []
        f = open(sadir, 'r')
        lines = [x.split() for x in f.readlines() if x[0:3] == 'ASG']
        f.close()
        secondary_last = 'C' # for unassigned residues.
        for i in range(len_df):
            atom_chain, atom_res, atom_het, atom_posid, atom_inode, atom_full_name, atom_name = df.iloc[i, :].values[0:7]
            atom_position = str(atom_het) + str(atom_posid) + str(atom_inode)
            atom_position = atom_position.strip()
            targetlst = list(filter(lambda x: x[2] == atom_chain and x[3] == atom_position, lines))
            if len(targetlst) > 0:
                target = targetlst[0]
                resname = aa_321dict[target[1][0] + target[1][1].lower() + target[1][2].lower()]
                secondary = target[5]
                phi, psi, sa = float(target[7]), float(target[8]), float(target[9])
                rsa = sa / ASA_dict[resname]
                if rsa > 1:
                    rsa = 1
                asa = ASA_dict[resname]
                secondarylst.append(secondary)
                salst.append(sa)
                rsalst.append(rsa)
                asalst.append(asa)
                philst.append(phi)
                psilst.append(psi)
                secondary_last = secondary
            else:
                secondarylst.append(secondary_last)
                resname = aa_321dict[atom_res]
                asa = ASA_dict[resname]
                salst.append(asa/2)
                rsalst.append(0.5)
                asalst.append(asa)
                philst.append(-71.05034549)# average of S2648
                psilst.append(69.74606526)# average of S2648

            # print('-'*10)
            # print(atom_chain)
            # print(atom_position)
            # print(atom_full_name,atom_name)
            # print(targetlst)
            # assert len(targetlst) == 1
            # target = targetlst[0]
            # resname = aa_321dict[target[1][0]+target[1][1].lower()+target[1][2].lower()]
            # secondary = target[5]
            # phi, psi, sa = float(target[7]), float(target[8]), float(target[9])
            # rsa = 100*sa/ASA_dict[resname]
            # if rsa > 100:
            #     rsa = 100
            # asa = ASA_dict[resname]
            # secondarylst.append(secondary)
            # salst.append(sa)
            # rsalst.append(rsa)
            # asalst.append(asa)
            # philst.append(phi)
            # psilst.append(psi)
        temp_df_sec = pd.DataFrame(np.array(secondarylst).reshape(len_df,1))
        temp_df_sa  = pd.DataFrame(np.array(salst).reshape(len_df, 1))
        temp_df_rsa = pd.DataFrame(np.array(rsalst).reshape(len_df, 1))
        temp_df_asa = pd.DataFrame(np.array(asalst).reshape(len_df, 1))
        temp_df_phi = pd.DataFrame(np.array(philst).reshape(len_df, 1))
        temp_df_psi = pd.DataFrame(np.array(psilst).reshape(len_df, 1))
        df.insert(7, 'secondary', temp_df_sec)
        ## OneHot encoding for Secondary Structure.
        temp_df = pd.DataFrame(np.zeros((len_df, 1)))
        ## Consider 7 types of secondary structure.
        df['s_H'] = temp_df
        df['s_G'] = temp_df
        df['s_I'] = temp_df
        df['s_E'] = temp_df
        df['s_B'] = temp_df
        df['s_T'] = temp_df
        df['s_C'] = temp_df
        df.loc[df.secondary == 'H', 's_H'] = 1
        df.loc[df.secondary == 'G', 's_G'] = 1
        df.loc[df.secondary == 'I', 's_I'] = 1
        df.loc[df.secondary == 'E', 's_E'] = 1
        df.loc[(df.secondary == 'B') | (df.secondary == 'b'), 's_B'] = 1
        df.loc[df.secondary == 'T', 's_T'] = 1
        df.loc[df.secondary == 'C', 's_C'] = 1
        ##consider 3 types of secondary structure (Helix, Strand, Coil), denoted by (H, S, C).
        df['s_Helix'] = temp_df
        df['s_Strand'] = temp_df
        df['s_Coil'] = temp_df# {'H':'H','G':'H','I':'H','E':'E','B':'C','T':'C','C':'C'}
        df.loc[df.secondary == 'H', 's_Helix'] = 1
        df.loc[df.secondary == 'G', 's_Helix'] = 1
        df.loc[df.secondary == 'I', 's_Helix'] = 1
        df.loc[df.secondary == 'E', 's_Strand'] = 1
        df.loc[(df.secondary == 'B') | (df.secondary == 'b'), 's_Coil'] = 1
        df.loc[df.secondary == 'T', 's_Coil'] = 1
        df.loc[df.secondary == 'C', 's_Coil'] = 1
        ## append sa and
        df['sa'] = temp_df_sa
        df['rsa'] = temp_df_rsa
        df['asa'] = temp_df_asa
        df['phi'] = temp_df_phi
        df['psi'] = temp_df_psi
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
        df.loc[(df.atom_name != 'C') & (df.atom_name != 'O') & (df.atom_name != 'N'), 'Other'] = 1
        return df

    if feature == 'pharm':
        atom_class = ['hydrophobic', 'positive', 'negative', 'neutral', 'acceptor', 'donor', 'aromatic', 'sulphur']
        pharmlst = []
        for i in range(len_df):
            atom_chain, atom_res, atom_het, atom_posid, atom_inode, atom_full_name, atom_name = df.iloc[i, :].values[0:7]
            aa_pharm_dict_tmp = aa_atom_pharm_dict[atom_res.upper()]
            try:
                pharmlst.append(aa_pharm_dict_tmp[atom_full_name])
            except:
                print('dataset_name: %s, pdb: %s, wtaa: %s, mtchain: %s, mtposition: %s, mtaa: %s, serial: %s'%(dataset_name, pdb, wtaa, mtchain, mtposition, mtaa, serial))
                print('atom full name: %s'%atom_full_name)
                pharmlst.append(aa_pharm_dict_tmp[atom_full_name[0]]) #for unassigned atom in xscore, such as OXT.
        pharm_df = pd.DataFrame(np.array(pharmlst).reshape(len_df,8), columns = atom_class)
        df = pd.concat([df, pharm_df], axis=1)
        return df

    if feature == 'hp':
        atom_class = ['hydrophobic_bak', 'polar']
        hplst = []
        for i in range(len_df):
            atom_chain, atom_res, atom_het, atom_posid, atom_inode, atom_full_name, atom_name = df.iloc[i, :].values[0:7]
            aa_hp_dict_tmp = aa_atom_hp_dict[atom_res.upper()]
            try:
                hplst.append(aa_hp_dict_tmp[atom_full_name])
            except:
                print('dataset_name: %s, pdb: %s, wtaa: %s, mtchain: %s, mtposition: %s, mtaa: %s, serial: %s' % (dataset_name, pdb, wtaa, mtchain, mtposition, mtaa, serial))
                print('atom full name: %s' % atom_full_name)
                hplst.append([0,0]) # for unassigned atom in xscore at HP classification.
        hp_df = pd.DataFrame(np.array(hplst).reshape(len_df, 2), columns=atom_class)
        df = pd.concat([df, hp_df], axis=1)
        return df

    if feature == 'mass':
        temp_df = pd.DataFrame(np.zeros((len_df, 1)))
        df['C_mass'] = temp_df
        df['O_mass'] = temp_df
        df['N_mass'] = temp_df
        df['S_mass'] = temp_df
        df.loc[df.atom_name == 'C', 'C_mass'] = aa_atom_mass['C']
        df.loc[df.atom_name == 'O', 'O_mass'] = aa_atom_mass['O']
        df.loc[df.atom_name == 'N', 'N_mass'] = aa_atom_mass['N']
        df.loc[df.atom_name == 'S', 'S_mass'] = aa_atom_mass['S']
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

    if feature == 'pharm_deltar':
        pharm_delta_r = np.array(aa_pharm_dict[mtaa]) - np.array(aa_pharm_dict[wtaa])
        temp_df = pd.DataFrame(np.ones((len_df, 1)))
        df['dhydrophobic'] = temp_df * pharm_delta_r[0]
        df['dpositive']    = temp_df * pharm_delta_r[1]
        df['dnegative']    = temp_df * pharm_delta_r[2]
        df['dneutral']     = temp_df * pharm_delta_r[3]
        df['dacceptor']    = temp_df * pharm_delta_r[4]
        df['ddonor']       = temp_df * pharm_delta_r[5]
        df['daromatic']    = temp_df * pharm_delta_r[6]
        df['dsulphur']     = temp_df * pharm_delta_r[7]
        return df

    if feature == 'hp_deltar':
        hp_delta_r = np.array(aa_hp_dict[mtaa]) - np.array(aa_hp_dict[wtaa])
        temp_df = pd.DataFrame(np.ones((len_df, 1)))
        df['dhydrophobic_bak'] = temp_df * hp_delta_r[0]
        df['dpolar'] = temp_df * hp_delta_r[1]
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
    parser.add_argument('-f', '--feature',   nargs='*',  type=str,    choices=['rsa','thermo','onehot','pharm', 'hp', 'mass','deltar','pharm_deltar','hp_deltar', 'msa','ddg'], default='', help='The feature to append, default=""')
    parser.add_argument('-S', '--sadir',     type=str,   help='The SA output directory of this pdb file')
    parser.add_argument('-t', '--thermo',    nargs=2,    type=float,  help='The pH and Temperature value to append')
    parser.add_argument('-d', '--ddg',       type=str,   help='The DDG value to append')

    args     = parser.parse_args()
    CSVDIR   = args.csvdir
    FILENAME = args.filename
    if args.outdir:
        OUTDIR = args.outdir
        if not os.path.exists(OUTDIR):
            os.mkdir(OUTDIR)
    if args.sadir:
        SADIR = args.sadir
    if args.thermo:
        THERMO = args.thermo
    if args.ddg:
        DDG = float(args.ddg)
        # print('ddg get')

    if not args.feature:
        print('Nothing to do!')
    else:
        f = open(CSVDIR, 'r')
        df = pd.read_csv(f)
        f.close()
        for FEATURE in args.feature:
            df = append_feature(df, FEATURE, FILENAME, THERMO[0], THERMO[1], DDG, SADIR)

        save_csv(df, FILENAME, OUTDIR=OUTDIR)