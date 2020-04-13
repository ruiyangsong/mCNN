#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
** Script for Calculate mCNN features (for each mutation). which was revoked by run_coord.py for "qsub" action on cluster.
   i.e., integrating all the features to specific (k-neighboring) csv file.

** NOTICE
   1. Both rosetta ref and rosetta mut are considered. It should be pointed out that all the items are based on those
      successful runs of rosetta_[ref|mut], i.e., those failed items (rosetta runs failed) were dropped.
   2. Other spatial feature such as orientation, sine or cosine values of dihedral angles, etc. can be calculated by coords in the csv file.
   3. left for blank

** File Name: coord.py rather than mCNN.py for the reason that it is contradict-free with the self-defined module named mCNN.
   The self-defined module mCNN was implemented by the following mapping:
   <1> Create the mapping: /public/home/sry/opt/miniconda3/envs/bio/custom_path/mCNN -> /public/home/sry/projects/mCNN/src/.
   <2> Adding pth file: /public/home/sry/opt/miniconda3/envs/bio/lib/python3.6/site-packages/custom_path.pth
       $ cat custom_path.pth with the output: /public/home/sry/opt/miniconda3/envs/bio/custom_path

** 10/10/2019.
** --sry.
'''

import os, sys,argparse
import numpy as np
import pandas as pd
from mCNN.processing import aa_321dict,log,read_csv,str2bool,aa_123dict

def main_all_atom():
    '''
    计算native_wild，TR_wild, TR_mutant的所有CA原子并保存在csv文件中
    :return:
    '''
    csvpth = '/public/home/sry/mCNN/dataset/TR/S2648_TR500.csv'
    df = pd.read_csv(csvpth)
    for i in range(len(df)):
        key, PDB, WILD_TYPE, CHAIN, POSITION, MUTANT, PH, TEMPERATURE, DDG = df.iloc[i, :].values
        mutant_tag = '%s.%s.%s.%s.%s.%s' % (key, PDB, WILD_TYPE, CHAIN, POSITION, MUTANT)
        ## for wild
        outdir = '/public/home/sry/mCNN/dataset/TR/feature/coord/wild'
        pdbpth = '/public/home/sry/mCNN/dataset/TR/pdb_chain/%s.pdb'%PDB
        stridepth = '/public/home/sry/mCNN/dataset/TR/feature/stride/wild/%s.stride'%PDB

        df_pdb, center_coord = ParsePDB(pdbpth, mutant_tag, accept_atom=('CA',), center='CA')
        FG = FeatureGenerator()
        df_feature = FG.append_stride(df_pdb=df_pdb,stride_pth=stridepth)
        save_csv(df_feature, outdir=outdir, filename='%s_neighbor_all'%PDB)

        ## for TR output
        outdir = '/public/home/sry/mCNN/dataset/TR/feature/coord/TR'
        # TR_wild
        TR_wild_tag = '%s.%s.%s' % (key, PDB, CHAIN)
        pdbpth = '/public/home/sry/mCNN/dataset/TR/output/%s/model1.pdb' % TR_wild_tag
        stridepth = '/public/home/sry/mCNN/dataset/TR/feature/stride/TR/%s.stride' % TR_wild_tag
        df_pdb, center_coord = ParsePDB(pdbpth, mutant_tag, accept_atom=('CA',), center='CA')
        FG = FeatureGenerator()
        df_feature = FG.append_stride(df_pdb=df_pdb, stride_pth=stridepth)
        save_csv(df_feature, outdir=outdir, filename='%s_neighbor_all' % TR_wild_tag)
        # TR_mut
        pdbpth = '/public/home/sry/mCNN/dataset/TR/output/%s/model1.pdb' % mutant_tag
        stridepth = '/public/home/sry/mCNN/dataset/TR/feature/stride/TR/%s.stride' % mutant_tag
        df_pdb, center_coord = ParsePDB(pdbpth, mutant_tag, accept_atom=('CA',), center='CA')
        FG = FeatureGenerator()
        df_feature = FG.append_stride(df_pdb=df_pdb,stride_pth=stridepth)
        save_csv(df_feature, outdir=outdir, filename='%s_neighbor_all' % mutant_tag)

def main_appending_wild_TR():
    '''将native_wild的原子append到TR_mutant后面（原子之间相互对应，即对应的残基是一样的）'''
    kneighbor = 20
    csvpth = '/public/home/sry/mCNN/dataset/TR/S2648_TR500.csv'
    outdir = '/public/home/sry/mCNN/dataset/TR/feature/coord/wild_TR'
    df = pd.read_csv(csvpth)
    for i in range(len(df)):
        key, PDB, WILD_TYPE, CHAIN, POSITION, MUTANT, PH, TEMPERATURE, DDG = df.iloc[i, :].values
        mutant_tag = '%s.%s.%s.%s.%s.%s' % (key, PDB, WILD_TYPE, CHAIN, POSITION, MUTANT)
        csvpth1 = '/public/home/sry/mCNN/dataset/TR/feature/coord/wild/%s_neighbor_all.csv'%PDB
        csvpth2 = '/public/home/sry/mCNN/dataset/TR/feature/coord/TR/%s_neighbor_all.csv'%mutant_tag
        df_neighbor = get_corresponding_coord_wild_TR(csvpth1, csvpth2, mutant_tag, kneighbor=kneighbor)
        save_csv(df_neighbor,outdir=outdir,filename='%s_neighbor_%s' % (mutant_tag,kneighbor))

def main_appending_TR_TR():
    '''将TR_wild的原子append到TR_mutant后面（原子之间相互对应，即对应的残基是一样的）'''
    kneighbor = 20
    csvpth = '/public/home/sry/mCNN/dataset/TR/S2648_TR500.csv'
    outdir = '/public/home/sry/mCNN/dataset/TR/feature/coord/TR_TR'
    df = pd.read_csv(csvpth)
    for i in range(len(df)):
        key, PDB, WILD_TYPE, CHAIN, POSITION, MUTANT, PH, TEMPERATURE, DDG = df.iloc[i, :].values
        mutant_tag = '%s.%s.%s.%s.%s.%s' % (key, PDB, WILD_TYPE, CHAIN, POSITION, MUTANT)
        wild_tag   = '%s.%s.%s'% (key, PDB, CHAIN)
        csvpth1 = '/public/home/sry/mCNN/dataset/TR/feature/coord/TR/%s_neighbor_all.csv'%wild_tag
        csvpth2 = '/public/home/sry/mCNN/dataset/TR/feature/coord/TR/%s_neighbor_all.csv'%mutant_tag
        df_neighbor = get_corresponding_coord_TR_TR(csvpth1, csvpth2, mutant_tag, kneighbor=kneighbor)
        save_csv(df_neighbor, outdir=outdir, filename='%s_neighbor_%s' % (mutant_tag, kneighbor))
# ----------------------------------------------------------------------------------------------------------------------

def save_csv(df,outdir,filename):
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    df.to_csv('%s/%s.csv'%(outdir,filename),index=False)

@log
def ParsePDB(pdbpth, mutant_tag, accept_atom = ('CA',), center='CA'):
    """
    :param pdbpth:
    :param mutant_tag:# ['key', 'PDB', 'WILD_TYPE', 'CHAIN', 'POSITION', 'MUTANT']
    :param atom_list:
    :param center:
    :return:
    """
    import warnings
    from Bio import BiopythonWarning
    from Bio.PDB.PDBParser import PDBParser
    warnings.simplefilter('ignore', BiopythonWarning)
    df_pdb = pd.DataFrame(
        {'chain': [], 'res': [], 'het': [], 'posid': [], 'inode': [], 'full_name': [], 'atom_name': [],
         'dist': [], 'x': [], 'y': [], 'z': [], 'occupancy': [], 'b_factor': []})
    key,pdbid,wtaa,mtchain,pos,mtaa = mutant_tag.split('.')
    print('The pdbid is:', pdbid, 'pth: %s' % pdbpth)
    # --------------------------------------------------------------------------------------------------------------
    # consider mapping
    if pdbpth.split('/')[-1] == 'model1.pdb':
        map_pos_pth = '/public/home/sry/mCNN/dataset/TR/map_pos/%s_mapping.csv'%pdbid
        df_map = pd.read_csv(map_pos_pth)
        df_map[['POSITION_OLD']] = df_map[['POSITION_OLD']].astype(str)
        df_map[['POSITION_NEW']] = df_map[['POSITION_NEW']].astype(str)

        pos = df_map.loc[(df_map.CHAIN == mtchain) & (df_map.POSITION_OLD == pos),'POSITION_NEW'].values[0]     #CHAIN,POSITION_OLD,POSITION_NEW
    # --------------------------------------------------------------------------------------------------------------

    if pos.isdigit():
        INODE = ' '
        POSID = int(pos)
    else:
        INODE = pos[-1]
        POSID = int(pos[:-1])
    MT_pos = (' ',POSID,INODE)

    parser = PDBParser(PERMISSIVE=1)
    structure = parser.get_structure(pdbid, pdbpth)
    model = structure[0]
    if pdbpth.split('/')[-1] == 'model1.pdb':
        try:
            assert model['A'][MT_pos].get_resname() == aa_123dict[wtaa]#TR_wild
        except:
            assert model['A'][MT_pos].get_resname() == aa_123dict[mtaa]#TR_mut
    else:
        assert model[mtchain][MT_pos].get_resname() == aa_123dict[wtaa]
    if center == 'CA':
        if pdbpth.split('/')[-1] == 'model1.pdb':
            center_coord = model['A'][MT_pos]['CA'].get_coord()
        else:
            center_coord = model[mtchain][MT_pos]['CA'].get_coord()

    for chain in model:
        chain_name = chain.get_id()
        res_id_lst = [res.get_id() for res in chain]

        print('The res_number in chain %s is: %d'%(chain_name,len(res_id_lst)))

        res_list = [chain[res_id] for res_id in res_id_lst]
        for res in res_list:
            res_name = res.get_resname()
            het, pos_id, inode = res.get_id()
            for atom in res:
                full_name, coord, occupancy, b_factor = atom.get_name(), atom.get_coord(), atom.get_occupancy(), atom.get_bfactor()
                if not full_name in accept_atom:
                    continue
                name = full_name.strip()[0]
                # if name in ('0','1','2','3','4','5','6','7','8','9','H','D'):
                # if not name in ('C','O','N','S'):
                dist = np.linalg.norm(center_coord - coord)
                x,y,z = coord
                temp_array = np.array([chain_name,res_name,het,pos_id,inode,full_name,name,dist,x,y,z,occupancy,b_factor]).reshape(1, -1)
                temp_df = pd.DataFrame(temp_array)
                temp_df.columns = df_pdb.columns
                df_pdb = pd.concat([df_pdb, temp_df], axis=0, ignore_index=True)
                break
    df_pdb[['dist']] = df_pdb[['dist']].astype(float)
    print('The atom_number (only CA) is:',len(df_pdb))
    return df_pdb, center_coord

def get_corresponding_coord_TR_TR(csvpth1, csvpth2, mutant_tag, kneighbor=20):
    print(csvpth1)
    print(csvpth2)
    success_cnt = 0
    df_lst = []
    key, PDB, WILD_TYPE, CHAIN, POSITION, MUTANT = mutant_tag.split('.')
    df1 = pd.read_csv(csvpth1)
    df2 = pd.read_csv(csvpth2)

    for i in range(len(df1)):
        try:
            assert df1.iloc[i][['res']].values[0] == df2.iloc[i][['res']].values[0]
        except:
            # print(df1.iloc[i][['res']].values[0],df2.iloc[i][['res']].values[0])
            assert df1.iloc[i][['res']].values[0] == aa_123dict[WILD_TYPE] and df2.iloc[i][['res']].values[0] == aa_123dict[MUTANT]

    df2_neighbor = CalNeighbor(df2, k_neighbor=kneighbor)
    df2_neighbor.reset_index(drop=True, inplace=True)
    for i in range(len(df2_neighbor)):
        if success_cnt == kneighbor:
            break
        res_df2, het_df2, posid_df2, inode_df2 = df2_neighbor.iloc[i][['res', 'het', 'posid', 'inode']].values
        try:
            dist_wild, x_wild, y_wild, z_wild, s_Helix_wild, s_Strand_wild, s_Coil_wild, sa_wild, rsa_wild, asa_wild, phi_wild, psi_wild = \
                df1.loc[(df1.het == het_df2) & (df1.posid == posid_df2) & (df1.inode == inode_df2),
                        ['dist', 'x', 'y', 'z', 's_Helix', 's_Strand', 's_Coil', 'sa', 'rsa', 'asa', 'phi', 'psi']].values[0]
            df_lst.append([dist_wild, x_wild, y_wild, z_wild, s_Helix_wild, s_Strand_wild, s_Coil_wild, sa_wild, rsa_wild, asa_wild, phi_wild, psi_wild])
            success_cnt += 1
        except:
            print('_' * 100)
            print('[ERROR at mut_site] posid_df2: %s' % (posid_df2))
            sys.exit(1)
    temp_df = pd.DataFrame(np.array(df_lst),
                           columns=['dist_wild', 'x_wild', 'y_wild', 'z_wild', 's_Helix_wild', 's_Strand_wild',
                                    's_Coil_wild', 'sa_wild', 'rsa_wild', 'asa_wild', 'phi_wild', 'psi_wild'])
    temp_df.columns=['dist_wild', 'x_wild', 'y_wild', 'z_wild', 's_Helix_wild', 's_Strand_wild','s_Coil_wild', 'sa_wild', 'rsa_wild', 'asa_wild', 'phi_wild', 'psi_wild']
    # print(temp_df)
    df_neighbor = pd.concat([df2_neighbor, temp_df], axis=1)
    return df_neighbor


def get_corresponding_coord_wild_TR(csvpth1, csvpth2, mutant_tag, kneighbor=20):
    """
    从csvpth2中在csvpth1中反向查找对应的 CA 原子，注意：wild_structure 中有的残基可能没有解析出 alpha-C 原子.
    csv columns are: [chain,res,het,posid,inode,full_name,atom_name,secondary,dist,x,y,z,occupancy,b_factor,s_Helix,s_Strand,s_Coil,sa,rsa,asa,phi,psi]
    :param csvpth1: wild_structure csv with all CA atoms.
    :param csvpth2: mut_structure csv with all CA atoms.
                    e.g., /public/home/sry/mCNN/dataset/TR/feature/coord/TR/976.1EY0.I.A.139.G_neighbor_all.csv
    :return:
    """
    print(csvpth1)
    print(csvpth2)
    key, PDB, WILD_TYPE, CHAIN, POSITION, MUTANT = mutant_tag.split('.')
    map_pos_pth = '/public/home/sry/mCNN/dataset/TR/map_pos/%s_mapping.csv'%PDB
    df1 = pd.read_csv(csvpth1)
    df2 = pd.read_csv(csvpth2)
    df_map = pd.read_csv(map_pos_pth)
    # print(df1.dtypes)
    # print(df2.dtypes)
    # print(df_map.dtypes)
    df_map[['POSITION_OLD']] = df_map[['POSITION_OLD']].astype(str)
    df_map[['POSITION_NEW']] = df_map[['POSITION_NEW']].astype(str)

    df2_neighbor = CalNeighbor(df2,k_neighbor=kneighbor+10)#有的res在wild里面找不到对应的，此时同时删除，故neighbor选多10个
    df_neighbor = CalNeighbor(df2,k_neighbor=kneighbor+10)
    df2_neighbor.reset_index(drop=True, inplace=True)
    df_neighbor.reset_index(drop=True, inplace=True)
    success_cnt = 0
    df_lst = []

    try:
        INODE = ' '
        POSID = int(POSITION)
    except:
        print(POSITION)
        INODE = POSITION[-1]
        POSID = int(POSITION[:-1])

    for i in range(len(df2_neighbor)):
        res_df2, het_df2, posid_df2, inode_df2 = df2_neighbor.iloc[i][['res', 'het', 'posid', 'inode']].values
        pos_df2 = (str(het_df2)+str(posid_df2)+str(inode_df2)).strip()
        pos_df1 = df_map.loc[df_map.POSITION_NEW == pos_df2, 'POSITION_OLD'].values[0]
        try:
            inode_df1 = ' '
            posid_df1 = int(pos_df1)
        except:
            print(pos_df1)
            inode_df1 = pos_df1[-1]
            posid_df1 = int(pos_df1[:-1])

        if pos_df1 == POSITION:
            try:
                dist_wild,x_wild,y_wild,z_wild,s_Helix_wild,s_Strand_wild,s_Coil_wild,sa_wild,rsa_wild,asa_wild,phi_wild,psi_wild = \
                    df1.loc[(df1.res==aa_123dict[WILD_TYPE])&(df1.posid==POSID)&(df1.inode==INODE),
                            ['dist','x','y','z','s_Helix','s_Strand','s_Coil','sa','rsa','asa','phi','psi']].values[0]
                df_lst.append([dist_wild,x_wild,y_wild,z_wild,s_Helix_wild,s_Strand_wild,s_Coil_wild,sa_wild,rsa_wild,asa_wild,phi_wild,psi_wild])
                success_cnt+=1
            except:
                print('_' * 100)
                print('[ERROR at mut_site] pos_df1: %s, pos_df2: %s' % (pos_df1, pos_df2))
                sys.exit(1)
        else:
            try:
                dist_wild,x_wild,y_wild,z_wild,s_Helix_wild,s_Strand_wild,s_Coil_wild,sa_wild,rsa_wild,asa_wild,phi_wild,psi_wild = \
                    df1.loc[(df1.res == res_df2) & (df1.posid == posid_df1)&(df1.inode==inode_df1),
                            ['dist','x','y','z','s_Helix','s_Strand','s_Coil','sa','rsa','asa','phi','psi']].values[0]
                df_lst.append([dist_wild,x_wild,y_wild,z_wild,s_Helix_wild,s_Strand_wild,s_Coil_wild,sa_wild,rsa_wild,asa_wild,phi_wild,psi_wild])
                success_cnt+=1
            except:
                print('_'*100)
                print('[ERROR] pos_df1: %s, pos_df2: %s'%(pos_df1,pos_df2))
                df_neighbor.drop(index=i,inplace=True)
                df_neighbor.reset_index(drop=True, inplace=True)
                continue

    temp_df = pd.DataFrame(np.array(df_lst), columns=['dist_wild','x_wild','y_wild','z_wild','s_Helix_wild','s_Strand_wild','s_Coil_wild','sa_wild','rsa_wild','asa_wild','phi_wild','psi_wild'])
    df_neighbor = pd.concat([df_neighbor, temp_df], axis=1)
    df_neighbor = CalNeighbor(df_neighbor, k_neighbor=kneighbor)
    return df_neighbor

@log
def CalNeighbor(df, k_neighbor=20):
    print('The k_number number is: %s'%k_neighbor)
    dist_arr = df.loc[:,'dist'].values
    assert len(dist_arr) >= k_neighbor
    indices  = sorted(dist_arr.argsort()[:k_neighbor])
    df_neighbor = df.iloc[indices,:]
    return df_neighbor

class FeatureGenerator(object):
    def __init__(self):
        self.init_constant()
    def init_constant(self):

        self.stride_secondary = {'H':'H',
                                 'G':'H',
                                 'I':'H',
                                 'E':'E',
                                 'B':'C',
                                 'T':'C',
                                 'C':'C'} #https://ssbio.readthedocs.io/en/latest/instructions/stride.html

        self.ASA_dict = {'A': 110.2, 'C': 140.4, 'D': 144.1, 'E': 174.7, 'F': 200.7,
                         'G': 78.7,  'H': 181.9, 'I': 185.0, 'K': 205.7, 'L': 183.1,
                         'M': 200.1, 'N': 146.4, 'P': 141.9, 'Q': 178.6, 'R': 229.0,
                         'S': 117.2, 'T': 138.7, 'V': 153.7, 'W': 240.5, 'Y': 213.7}

        self.aa_321dict = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
                           'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
                           'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
                           'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'}  # from wiki

        self.aa_atom_dict = {'A': {'C': 3,  'H': 7,  'O': 2, 'N': 1},
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

        self.aa_pharm_dict = {'A':[1,0,0,5,1,1,0,0],  'R':[2,2,0,9,1,4,0,0],  'N':[1,0,0,8,2,2,0,0],  'D':[1,0,2,6,3,1,0,0],
                              'C':[2,0,0,6,1,1,0,1],  'Q':[2,0,0,9,2,2,0,0],  'E':[2,0,2,7,3,1,0,0],  'G':[0,0,0,4,1,1,0,0],
                              'H':[1,2,0,8,3,3,5,0],  'I':[4,0,0,8,1,1,0,0],  'L':[4,0,0,8,1,1,0,0],  'K':[3,1,0,8,1,2,0,0],
                              'M':[4,0,0,8,1,1,0,1],  'F':[7,0,0,11,1,1,6,0], 'P':[2,0,0,7,1,0,0,0],  'S':[0,0,0,6,2,2,0,0],
                              'T':[1,0,0,7,2,2,0,0],  'W':[7,0,0,14,1,2,9,0], 'Y':[6,0,0,12,2,2,6,0], 'V':[3,0,0,7,1,1,0,0]}

        self.aa_hp_dict = {'A':[1,2], 'R':[2,4], 'N':[1,3], 'D':[1,3], 'C':[2,2], 'Q':[2,3], 'E':[2,3], 'G':[0,2], 'H':[1,5], 'I':[4,2],
                           'L':[4,2], 'K':[3,3], 'M':[4,2], 'F':[7,2], 'P':[2,4], 'S':[0,3], 'T':[1,3], 'W':[7,4], 'Y':[6,3], 'V':[3,2]}

        self.aa_vec_dict = {}  # aa_vec_dict = {'aa_name':[vec_of_atom_number by class], ...}
        ## Calc aa_vec_dict.
        for aa_name in self.aa_atom_dict.keys():
            aa_vec = list(self.aa_atom_dict[aa_name].values())
            if len(aa_vec) == 4:
                aa_vec.append(0)
            self.aa_vec_dict[aa_name] = aa_vec

    def calEntropy(self, filedir, position):
        '''
        return frq vector at position and entropy of this frq vector.
        '''
        data = np.load(filedir, allow_pickle=True)
        frq = data['frq']
        position_index = frq[1:, 1]
        index = np.argwhere(position_index == str(position)) + 1
        rv = frq[index, 3:].astype(float).reshape(-1)
        rv_pop = list(filter(lambda x: x != 0, rv))
        entropy = sum(list(map(lambda x: -x * np.log2(x), rv_pop)))
        return rv, entropy

    def random_dihedral(self):
        r = np.random.random()
        if (r <= 0.135):
            phi = -140
            psi = 153
        elif (r > 0.135 and r <= 0.29):
            phi = -72
            psi = 145
        elif (r > 0.29 and r <= 0.363):
            phi = -122
            psi = 117
        elif (r > 0.363 and r <= 0.485):
            phi = -82
            psi = -14
        elif (r > 0.485 and r <= 0.982):
            phi = -61
            psi = -41
        else:
            phi = 57
            psi = 39
        return (phi, psi)

    @log
    def append_stride(self,df_pdb,stride_pth):
        df_pdb.reset_index(drop=True, inplace=True)
        len_df = len(df_pdb)
        print('<--->appending stride features')
        ## RSA, residue oriented
        secondarylst = []
        salst = []
        rsalst = []
        asalst = []
        philst = []
        psilst = []
        with open(stride_pth, 'r') as f:
            lines = [x.split() for x in f.readlines() if x[0:3] == 'ASG']
        secondary_last = 'C'  # for unassigned residues.
        for i in range(len_df):
            atom_chain, atom_res, atom_het, atom_posid, atom_inode, atom_full_name, atom_name = df_pdb.iloc[i,:].values[0:7]
            atom_position = (str(atom_het) + str(atom_posid) + str(atom_inode)).strip()
            targetlst = list(filter(lambda x: x[2] == atom_chain and x[3] == atom_position, lines))
            if len(targetlst) > 0:
                target = targetlst[0]
                resname = aa_321dict[target[1]]
                secondary = target[5]
                phi, psi, sa = float(target[7]), float(target[8]), float(target[9])
                rsa = sa / self.ASA_dict[resname]
                if rsa > 1:
                    rsa = 1
                asa = self.ASA_dict[resname]
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
                asa = self.ASA_dict[resname]
                salst.append(asa / 2)
                rsalst.append(0.5)
                asalst.append(asa)
                phi, psi = self.random_dihedral()
                philst.append(phi)
                psilst.append(psi)
        temp_df_sec = pd.DataFrame(np.array(secondarylst).reshape(len_df, 1))
        temp_df_sa = pd.DataFrame(np.array(salst).reshape(len_df, 1))
        temp_df_rsa = pd.DataFrame(np.array(rsalst).reshape(len_df, 1))
        temp_df_asa = pd.DataFrame(np.array(asalst).reshape(len_df, 1))
        temp_df_phi = pd.DataFrame(np.array(philst).reshape(len_df, 1))
        temp_df_psi = pd.DataFrame(np.array(psilst).reshape(len_df, 1))
        df_pdb.insert(7, 'secondary', temp_df_sec)
        ##consider 3 types of secondary structure (Helix, Strand, Coil), denoted by (H, S, C).
        # {'H':'H','G':'H','I':'H','E':'E','B':'C','T':'C','C':'C'}
        temp_df = pd.DataFrame(np.zeros((len_df, 3)), columns=['s_Helix', 's_Strand', 's_Coil'])
        df_pdb = pd.concat([df_pdb, temp_df], axis=1)

        df_pdb.loc[df_pdb.secondary == 'H', 's_Helix'] = 1
        df_pdb.loc[df_pdb.secondary == 'G', 's_Helix'] = 1
        df_pdb.loc[df_pdb.secondary == 'I', 's_Helix'] = 1
        df_pdb.loc[df_pdb.secondary == 'E', 's_Strand'] = 1
        df_pdb.loc[(df_pdb.secondary == 'B') | (df_pdb.secondary == 'b'), 's_Coil'] = 1
        df_pdb.loc[df_pdb.secondary == 'T', 's_Coil'] = 1
        df_pdb.loc[df_pdb.secondary == 'C', 's_Coil'] = 1

        ## append sa and...
        df_pdb['sa'] = temp_df_sa
        df_pdb['rsa'] = temp_df_rsa
        df_pdb['asa'] = temp_df_asa
        df_pdb['phi'] = temp_df_phi
        df_pdb['psi'] = temp_df_psi

        return df_pdb

if __name__ == '__main__':
    # main_all_atom()
    # main_appending_wild_TR()
    main_appending_TR_TR()