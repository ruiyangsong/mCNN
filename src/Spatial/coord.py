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

import os, argparse
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from Bio.PDB.HSExposure import HSExposureCA
from Bio.PDB.vectors import Vector, calc_dihedral, calc_angle
from mCNN.processing import PDBparser,aa_321dict,log,read_csv,str2bool


def main():
    # print(os.getcwd())
    ## Set params
    parser = argparse.ArgumentParser()
    parser.add_argument('--flag', type = str, choices=['first','all'],  default='all', help='"first" calc df_feature only, "all" calc df_neighbor and df_feature, default is "all".')
    # -----------------------------parameters for NeighborCalculator----------------------------------------------------
    parser.add_argument('-p', '--pdbdir',     type=str,   required=True,   help='the input pdb file directory')
    parser.add_argument('-tag', '--mutant_tag', type=str, required=True,   help='the mutant primary key of one pdb (do not considr pos_new eg.: 1VQB_V_A_70_C)')
    parser.add_argument('-k', '--k_neighbor', type=int,   required=True,   help='one value of k_neighbor.')
    parser.add_argument('-c', '--center',     type=str,   choices=['CA', 'geometric'], default='CA', help='One value of the type of MT site center.')
    # -----------------------------parameters for FeatureGenerator------------------------------------------------------
    parser.add_argument('-o', '--outdir',     type=str,   required=True,   help='The output directory')#~/mCNN/dataset/S2648/feature/mCNN/mutant/csv/1EY0_I_A_92_M
    parser.add_argument('-n', '--filename',   type=str,   required=True,   help='The name of output csv file')
    parser.add_argument('--reverse',          type=str,   required=True,   default='True', help='if reverse the mutant2wild(consider the reverse mutation, required for features which describe mutation, such as \Delta_r, ddg and etc.)')
    parser.add_argument('-f', '--feature',    type=str,   required=True,   nargs='+',
                        choices=['depth', 'rsa', 'thermo', 'onehot', 'pharm', 'hp', 'mass', 'deltar', 'pharm_deltar','hp_deltar', 'msa', 'energy', 'ddg'],
                        help='The feature to append, default=""')
    parser.add_argument('--wtblastdir',       type=str,   required=True,   help='wt blast dir')#without chain!! eg. ~/mCNN/dataset/S2648/feature/msa/mdl0_wild/1A43
    parser.add_argument('--mtblastdir',       type=str,   required=True,   help='mt blast dir')
    parser.add_argument('--energydir',        type=str,   required=True,   help='energy table dir')
    parser.add_argument('--mappingdir',       type=str,   required=True,   help='pos mapping csv file dir')
    parser.add_argument('--surfacedir',       type=str,   required=True,   help='The surface output directory of this pdb file')
    parser.add_argument('-S', '--sadir',      type=str,   required=True,   help='The SA output directory of this pdb file')
    parser.add_argument('-t', '--thermo',     type=str,   required=True,   nargs=2,         help='The pH and Temperature value to append')
    parser.add_argument('-d', '--ddg',        type=str,   required=True,   help='The DDG value to append')
    ## Parse params
    args = parser.parse_args()
    flag = args.flag
    # -----------------------------parameters for NeighborCalculator----------------------------------------------------
    pdbdir     = args.pdbdir
    mutant_tag = args.mutant_tag
    k_neighbor = args.k_neighbor
    center = args.center
    # -----------------------------parameters for FeatureGenerator------------------------------------------------------
    OUTDIR     = args.outdir
    if not os.path.exists(OUTDIR):
        os.mkdir(OUTDIR)
    FILENAME   = args.filename
    REVERSE    = str2bool(args.reverse)
    featurelst = args.feature
    WTBLASTDIR = args.wtblastdir
    MTBLASTDIR = args.mtblastdir
    ENERGYDIR  = args.energydir
    MAPPINGDIR = args.mappingdir
    SURFACEDIR = args.surfacedir
    SADIR      = args.sadir
    THERMO     = [float(x) for x in args.thermo]
    DDG        = float(args.ddg)
    ####################################################################################################################
    ## main program BEGINs here
    ####################################################################################################################
    NC = NeighborCalculator(pdbdir,mutant_tag,k_neighbor,center)

    if flag == 'first' and not os.path.exists('%s/center_%s.csv' % (OUTDIR, center)):
        ## cals df_feature (ALL the ATOMS) only.
        df_pdb, center_coord = NC.ParsePDB()
        FG = FeatureGenerator(df_pdb, mutant_tag, OUTDIR, FILENAME, featurelst, THERMO[0], THERMO[1], DDG, SADIR, WTBLASTDIR, MTBLASTDIR, ENERGYDIR, MAPPINGDIR, SURFACEDIR, REVERSE)
        df_feature = FG.append_feature()
        save_csv(df=df_feature, outdir=OUTDIR, filename='center_%s' % center)
        exit(0)

    elif flag == 'all':
        ## calc df_feature and df_neighbor.
        df_pdb, center_coord = NC.ParsePDB()

        ## 如果包含全部原子的特征文件不存在，则计算并保存它，以后可以基于此直接筛选近邻。
        if not os.path.exists('%s/center_%s.csv'%(OUTDIR,center)):
            FG = FeatureGenerator(df_pdb, mutant_tag, OUTDIR, FILENAME, featurelst, THERMO[0], THERMO[1], DDG, SADIR, WTBLASTDIR, MTBLASTDIR, ENERGYDIR, MAPPINGDIR, SURFACEDIR, REVERSE)
            df_feature = FG.append_feature()
            save_csv(df=df_feature, outdir=OUTDIR, filename='center_%s' % center)
            param_acider(NC, df_feature, OUTDIR, FILENAME, center, center_coord)
        ## 如果包含全部原子的文件已经存在，且目标近邻数的csv文件不存在，则读取包含全部原子的文件并筛选近邻。
        elif not os.path.exists('%s/%s.csv'%(OUTDIR,FILENAME)):
            df_feature = read_csv('%s/center_%s.csv'%(OUTDIR,center))
            param_acider(NC, df_feature, OUTDIR, FILENAME, center, center_coord)

        # ==============================================================================================================
        ## This block was replaced by function param_acider(NC, df_feature,OUTDIR,FILENAME,center,center_coord)
        # ==============================================================================================================
        # ## calc df_neighbor based on df_feature.
        # df_neighbor = NC.CalNeighbor(df_feature)
        # df_neighbor.reset_index(drop=True, inplace=True)
        # save_csv(df=df_neighbor,outdir=OUTDIR,filename=FILENAME)
        # ## save center_coord
        # if center == 'geometric':
        #     np.save('%s/%s_center_coord.npy' % (OUTDIR, FILENAME), center_coord)
        # if center == 'CA':
        #     if not os.path.exists('%s/center_CA_neighbor_._center_coord.npy' % OUTDIR):
        #         np.save('%s/center_CA_neighbor_._center_coord.npy' % OUTDIR, center_coord)

    ####################################################################################################################
    ## main program ENDs here
    ####################################################################################################################
def param_acider(NC, df_feature,OUTDIR,FILENAME,center,center_coord):
    '''this func helps to pass local variable "df_feature"'''
    ## calc df_neighbor based on df_feature.
    df_neighbor = NC.CalNeighbor(df_feature)
    df_neighbor.reset_index(drop=True, inplace=True)
    save_csv(df=df_neighbor, outdir=OUTDIR, filename=FILENAME)
    ## save center_coord
    if not os.path.exists('%s/center_%s_neighbor_._center_coord.npy' % (OUTDIR, center)):
        np.save('%s/center_%s_neighbor_._center_coord.npy' %(OUTDIR,center), center_coord)

def save_csv(df,outdir,filename):
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    df.to_csv('%s/%s.csv'%(outdir,filename),index=False)

class NeighborCalculator(object):
    def __init__(self,pdbdir,mutant_tag,k_neighbor,center):
        self.pdbdir       = pdbdir
        self.mutant_tag   = mutant_tag

        self.k_neighbor   = k_neighbor
        self.center       = center

    def CalcResDepth(self):
        '''Calc ResDepth by MSMS, CUZ some bug of the orginal program (*Unix version), this function left for blank'''
        pass

    def CalcHSE(self, model):
        exp_ca = HSExposureCA(model=model)
        return exp_ca.property_dict

    def CalcMainChainAVGCoord(self, res):
        atomcoordlst = [atom.get_coord() for atom in res if atom.get_name() in ('C', 'O', 'N', 'CA')]
        avg_coord = np.array([0., 0., 0.])
        for atomcoord in atomcoordlst:
            avg_coord += atomcoord
        return avg_coord / len(atomcoordlst)

    def CollectVector(self, res, atom_full_name):
        assert type(atom_full_name) is str
        try:
            vec = res[atom_full_name].get_vector()
        except:
            vec = Vector(self.CalcMainChainAVGCoord(res))
        return vec

    def Calc_Omega(self, res_query, res_base):
        '''[CA,CB] [_CA,_CB]'''
        CA_vec  = self.CollectVector(res=res_query,atom_full_name='CA')
        CB_vec  = self.CollectVector(res=res_query,atom_full_name='CB')
        _CA_vec = self.CollectVector(res=res_base,atom_full_name='CA')
        _CB_vec = self.CollectVector(res=res_base,atom_full_name='CB')
        return calc_dihedral(CA_vec, CB_vec, _CA_vec, _CB_vec)

    def Calc_theta(self,res_query, res_base):
        '''[CB,_CB] [_CA,_N]'''
        CB_vec  = self.CollectVector(res=res_query,atom_full_name='CB')
        _CB_vec = self.CollectVector(res=res_base,atom_full_name='CB')
        _CA_vec = self.CollectVector(res=res_base,atom_full_name='CA')
        _N_vec  = self.CollectVector(res=res_base,atom_full_name='N')
        return calc_dihedral(CB_vec, _CB_vec, _CA_vec, _N_vec)

    def Calc_phi(self,res_query, res_base):
        '''[CB, _CB, _CA]'''
        CB_vec  = self.CollectVector(res=res_query,atom_full_name='CB')
        _CB_vec = self.CollectVector(res=res_base,atom_full_name='CB')
        _CA_vec = self.CollectVector(res=res_base,atom_full_name='CA')
        return calc_angle(CB_vec, _CB_vec, _CA_vec)

    @log
    def ParsePDB(self):
        df_pdb = pd.DataFrame(
            {'chain': [], 'res': [], 'het': [], 'posid': [], 'inode': [], 'full_name': [], 'atom_name': [],
             'dist': [], 'omega_Orient': [], 'theta12': [], 'theta21': [], 'phi12': [], 'phi21': [], 'sin_omega': [],
             'cos_omega': [], 'sin_theta12': [], 'cos_theta12': [], 'sin_theta21': [], 'cos_theta21': [],
             'sin_phi12': [], 'cos_phi12': [], 'sin_phi21': [], 'cos_phi21': [], 'x': [], 'y': [], 'z': [],
             'x2CA': [], 'y2CA': [], 'z2CA': [], 'hse_up': [], 'hse_down': [], 'occupancy': [], 'b_factor': []}
        )

        pdbid,wtaa,mtchain,pos,mtaa = self.mutant_tag.split('_')
        model = PDBparser(pdbdir=self.pdbdir,MDL=0,write=0,outpath=None)

        if pos.isdigit():
            INODE = ' '
            POSID = int(pos)
        else:
            INODE = pos[-1]
            POSID = int(pos[:-1])
        MT_pos = (' ',POSID,INODE)

        if self.center == 'CA':
            center_coord = model[mtchain][MT_pos]['CA'].get_coord()
        elif self.center == 'geometric':
            # atomcoordlst = [atom.get_coord() for atom in model[mtchain][MT_pos] if not atom.get_name()[0] in ('0','1','2','3','4','5','6','7','8','9','H','D')]# do not consider H and D atoms.
            atomcoordlst = [atom.get_coord() for atom in model[mtchain][MT_pos] if atom.get_name()[0] in ('C','O','N','S')]
            center_coord = np.array([0., 0., 0.])
            for atomcoord in atomcoordlst:
                center_coord = center_coord + atomcoord
            center_coord = center_coord / len(atomcoordlst)

        print('The pdbid is:', pdbid)
        hse_dict = self.CalcHSE(model=model)
        for chain in model:
            chain_name = chain.get_id()
            res_id_lst = [res.get_id() for res in chain]

            print('The res_number in chain %s is: %d'%(chain_name,len(res_id_lst)))

            res_list = [chain[res_id] for res_id in res_id_lst]
            for res in res_list:
                res_name = res.get_resname()
                het, pos_id, inode = res.get_id()
                ## HSE
                hse_key = (chain_name,(het, pos_id, inode))
                if hse_key in hse_dict.keys():
                    hse_up, hse_down, angle = hse_dict[hse_key]
                else:
                    hse_up, hse_down, angle = 0, 0, 0
                ## Orient dihedral and angle
                omega   = self.Calc_Omega(res_query=res, res_base=model[mtchain][MT_pos])
                theta12 = self.Calc_theta(res_query=res, res_base=model[mtchain][MT_pos])
                theta21 = self.Calc_theta(res_query=model[mtchain][MT_pos], res_base=res)
                phi12   = self.Calc_phi(res_query=res, res_base=model[mtchain][MT_pos])
                phi21   = self.Calc_phi(res_query=model[mtchain][MT_pos], res_base=res)
                sin_omega   = np.sin(omega)
                cos_omega   = np.cos(omega)
                sin_theta12 = np.sin(theta12)
                cos_theta12 = np.cos(theta21)
                sin_theta21 = np.sin(theta21)
                cos_theta21 = np.cos(theta21)
                sin_phi12   = np.sin(phi12)
                cos_phi12   = np.cos(phi12)
                sin_phi21 = np.sin(phi21)
                cos_phi21 = np.cos(phi21)

                for atom in res:
                    full_name, coord, occupancy, b_factor = atom.get_name(), atom.get_coord(), atom.get_occupancy(), atom.get_bfactor()
                    name = full_name.strip()[0]
                    # if name in ('0','1','2','3','4','5','6','7','8','9','H','D'):
                    if not name in ('C','O','N','S'):
                        continue
                    dist = np.linalg.norm(center_coord - coord)
                    x,y,z = coord
                    x2CA, y2CA, z2CA = coord - center_coord ## orient vector

                    temp_array = np.array(
                        [chain_name, res_name, het, pos_id, inode, full_name, name, dist, omega, theta12, theta21, phi12,
                        phi21, sin_omega, cos_omega, sin_theta12, cos_theta12, sin_theta21, cos_theta21, sin_phi12, cos_phi12,
                        sin_phi21, cos_phi21, x, y, z, x2CA, y2CA, z2CA, hse_up, hse_down, occupancy, b_factor]
                    ).reshape(1, -1)

                    temp_df = pd.DataFrame(temp_array)
                    temp_df.columns = df_pdb.columns
                    df_pdb = pd.concat([df_pdb, temp_df], axis=0, ignore_index=True)
        print('The atom_number is:',len(df_pdb))
        return df_pdb,center_coord

    @log
    def CalNeighbor(self,df_feature):
        print('The k_number number is: %s'%self.k_neighbor)
        dist_arr = df_feature.loc[:,'dist'].values
        assert len(dist_arr) >= self.k_neighbor
        indices  = sorted(dist_arr.argsort()[:self.k_neighbor])
        df_neighbor = df_feature.iloc[indices,:]

        return df_neighbor

class FeatureGenerator(object):
    def __init__(self,df,mutant_tag,outdir,filename,feature_lst,pH,T,ddg,sadir,wtblastdir,mtblastdir,energydir,mappingdir,surfacedir,reverse_flag):
        self.df          = df
        self.mutant_tag  = mutant_tag
        self.feature_lst = feature_lst
        self.outdir      = outdir
        self.filename    = filename
        self.reverse     = reverse_flag
        self.pH          = pH
        self.T           = T
        self.ddg         = ddg
        self.sadir       = sadir
        self.wtblastdir  = wtblastdir
        self.mtblastdir  = mtblastdir
        self.energydir   = energydir
        self.mappingdir  = mappingdir
        self.surfacedir  = surfacedir
        self.init_constant()

    @log
    def init_constant(self):
        # self.energy_name_lst = ['fa_atr',              'fa_rep',      'fa_sol',    'fa_intra_rep',         'fa_intra_sol_xover4',
        #                         'lk_ball_wtd',         'fa_elec',     'pro_close', 'hbond_sr_bb',          'hbond_lr_bb',
        #                         'hbond_bb_sc',         'hbond_sc',    'dslf_fa13', 'atom_pair_constraint', 'angle_constraint',
        #                         'dihedral_constraint', 'omega',       'fa_dun',    'p_aa_pp',              'yhh_planarity',
        #                         'ref',                 'rama_prepro', 'total']

        self.energy_name_lst = ['fa_atr',              'fa_rep',      'fa_sol',    'fa_intra_rep',         'fa_intra_sol_xover4',
                                'lk_ball_wtd',         'fa_elec',     'pro_close',
                                'hbond_bb_sc',         'hbond_sc',
                                                       'omega',       'fa_dun',    'p_aa_pp',              'yhh_planarity',
                                'ref',                 'rama_prepro', 'total']

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

            # secondary_lst = ['G', 'H', 'b', 'C', 'T', 'B', 'E']
        self.aa_atom_mass = {'C': 12.0107, 'H': 1.0079, 'O': 15.9994, 'N': 14.0067, 'S': 32.065} # from https://www.lenntech.com/periodic/mass/atomic-mass.htm

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

        self.aa_atom_pharm_dict = {'ALA':
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

        self.aa_atom_hp_dict = {'ALA':
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

        self.aa_pharm_dict = {'A':[1,0,0,5,1,1,0,0],  'R':[2,2,0,9,1,4,0,0],  'N':[1,0,0,8,2,2,0,0],  'D':[1,0,2,6,3,1,0,0],
                              'C':[2,0,0,6,1,1,0,1],  'Q':[2,0,0,9,2,2,0,0],  'E':[2,0,2,7,3,1,0,0],  'G':[0,0,0,4,1,1,0,0],
                              'H':[1,2,0,8,3,3,5,0],  'I':[4,0,0,8,1,1,0,0],  'L':[4,0,0,8,1,1,0,0],  'K':[3,1,0,8,1,2,0,0],
                              'M':[4,0,0,8,1,1,0,1],  'F':[7,0,0,11,1,1,6,0], 'P':[2,0,0,7,1,0,0,0],  'S':[0,0,0,6,2,2,0,0],
                              'T':[1,0,0,7,2,2,0,0],  'W':[7,0,0,14,1,2,9,0], 'Y':[6,0,0,12,2,2,6,0], 'V':[3,0,0,7,1,1,0,0]}

        self.aa_hp_dict = {'A':[1,2], 'R':[2,4], 'N':[1,3], 'D':[1,3], 'C':[2,2], 'Q':[2,3], 'E':[2,3], 'G':[0,2], 'H':[1,5], 'I':[4,2],
                           'L':[4,2], 'K':[3,3], 'M':[4,2], 'F':[7,2], 'P':[2,4], 'S':[0,3], 'T':[1,3], 'W':[7,4], 'Y':[6,3], 'V':[3,2]}

        self.aa_vec_dict = {'A': [3, 7, 2, 1, 0], 'R': [6, 14, 2, 4, 0], 'N': [4, 8, 3, 2, 0], 'D': [4, 7, 4, 1, 0],
                            'C': [3, 7, 2, 1, 1], 'Q': [5, 10, 3, 2, 0], 'E': [5, 9, 4, 1, 0], 'G': [2, 5, 2, 1, 0],
                            'H': [6, 9, 2, 3, 0], 'I': [6, 13, 2, 1, 0], 'L': [6, 13, 2, 1, 0], 'K': [6, 14, 2, 2, 0],
                            'M': [5, 11, 2, 1, 1], 'F': [9, 11, 2, 1, 0], 'P': [5, 9, 2, 1, 0], 'S': [3, 7, 3, 1, 0],
                            'T': [4, 9, 3, 1, 0], 'W': [11, 12, 2, 2, 0], 'Y': [9, 11, 3, 1, 0], 'V': [5, 11, 2, 1, 0]}


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
    def append_feature(self):
        len_df = len(self.df)
        pdbid, wtaa, mtchain, mtpos, mtaa = self.mutant_tag.split('_')
        for feature in self.feature_lst:
            print('<--->appending %s'%feature)

            ## depth to protein surface, atom oriented
            if feature == 'depth':
                coord   = self.df.loc[:,['x','y','z']].values
                data = np.load(self.surfacedir)
                surface = data['surface']
                res = cdist(coord, surface, metric='euclidean')

                temp_df = pd.DataFrame(np.min(res, axis=1).reshape(len_df,-1),columns=['depth'])
                self.df = pd.concat([self.df, temp_df], axis=1)

            ## RSA, residue oriented
            if feature == 'rsa':
                secondarylst = []
                salst = []
                rsalst = []
                asalst = []
                philst = []
                psilst = []
                sin_philst = []
                cos_philst = []
                sin_psilst = []
                cos_psilst = []
                f = open(self.sadir, 'r')
                lines = [x.split() for x in f.readlines() if x[0:3] == 'ASG']
                f.close()
                secondary_last = 'C'  # for unassigned residues.
                for i in range(len_df):
                    atom_chain, atom_res, atom_het, atom_posid, atom_inode, atom_full_name, atom_name = self.df.iloc[i,:].values[0:7]
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
                        sin_philst.append(np.sin(phi))
                        cos_philst.append(np.cos(phi))
                        sin_psilst.append(np.sin(psi))
                        cos_psilst.append(np.cos(psi))
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
                        sin_philst.append(np.sin(phi))
                        cos_philst.append(np.cos(phi))
                        sin_psilst.append(np.sin(psi))
                        cos_psilst.append(np.cos(psi))

                temp_df_sec = pd.DataFrame(np.array(secondarylst).reshape(len_df, 1))
                temp_df_sa = pd.DataFrame(np.array(salst).reshape(len_df, 1))
                temp_df_rsa = pd.DataFrame(np.array(rsalst).reshape(len_df, 1))
                temp_df_asa = pd.DataFrame(np.array(asalst).reshape(len_df, 1))
                temp_df_phi = pd.DataFrame(np.array(philst).reshape(len_df, 1))
                temp_df_psi = pd.DataFrame(np.array(psilst).reshape(len_df, 1))
                temp_df_sin_phi = pd.DataFrame(np.array(sin_philst).reshape(len_df, 1))
                temp_df_cos_phi = pd.DataFrame(np.array(cos_philst).reshape(len_df, 1))
                temp_df_sin_psi = pd.DataFrame(np.array(sin_psilst).reshape(len_df, 1))
                temp_df_cos_psi = pd.DataFrame(np.array(cos_psilst).reshape(len_df, 1))
                self.df.insert(7, 'secondary', temp_df_sec)

                ## OneHot encoding for Secondary Structure.
                ## Consider 7 types of secondary structure.
                temp_df = pd.DataFrame(np.zeros((len_df,7)), columns=['s_H','s_G','s_I','s_E','s_B','s_T','s_C'])
                self.df = pd.concat([self.df, temp_df], axis=1)

                self.df.loc[self.df.secondary == 'H', 's_H'] = 1
                self.df.loc[self.df.secondary == 'G', 's_G'] = 1
                self.df.loc[self.df.secondary == 'I', 's_I'] = 1
                self.df.loc[self.df.secondary == 'E', 's_E'] = 1
                self.df.loc[(self.df.secondary == 'B') | (self.df.secondary == 'b'), 's_B'] = 1
                self.df.loc[self.df.secondary == 'T', 's_T'] = 1
                self.df.loc[self.df.secondary == 'C', 's_C'] = 1

                ##consider 3 types of secondary structure (Helix, Strand, Coil), denoted by (H, S, C).
                # {'H':'H','G':'H','I':'H','E':'E','B':'C','T':'C','C':'C'}
                temp_df = pd.DataFrame(np.zeros((len_df, 3)), columns=['s_Helix', 's_Strand', 's_Coil'])
                self.df = pd.concat([self.df, temp_df], axis=1)

                self.df.loc[self.df.secondary == 'H', 's_Helix'] = 1
                self.df.loc[self.df.secondary == 'G', 's_Helix'] = 1
                self.df.loc[self.df.secondary == 'I', 's_Helix'] = 1
                self.df.loc[self.df.secondary == 'E', 's_Strand'] = 1
                self.df.loc[(self.df.secondary == 'B') | (self.df.secondary == 'b'), 's_Coil'] = 1
                self.df.loc[self.df.secondary == 'T', 's_Coil'] = 1
                self.df.loc[self.df.secondary == 'C', 's_Coil'] = 1

                ## append sa and
                self.df['sa'] = temp_df_sa
                self.df['rsa'] = temp_df_rsa
                self.df['asa'] = temp_df_asa
                self.df['phi'] = temp_df_phi
                self.df['psi'] = temp_df_psi
                self.df['sin_phi'] = temp_df_sin_phi
                self.df['cos_phi'] = temp_df_cos_phi
                self.df['sin_psi'] = temp_df_sin_psi
                self.df['cos_psi'] = temp_df_cos_psi
            # ----------------------------------------------------------------------------------------------------------
            ## thermodynamic parameters(pH,temperature), atom oriented(each atom are the same).
            if feature == 'thermo':
                temp_df_ph = pd.DataFrame(np.ones((len_df, 1)) * self.pH)
                temp_df_t = pd.DataFrame(np.ones((len_df, 1)) * self.T)
                self.df['ph'] = temp_df_ph
                self.df['temperature'] = temp_df_t
                # print(self.df)
            # ----------------------------------------------------------------------------------------------------------
            ## onehot encoding vector.
            if feature == 'onehot':
                # atom level (C O N S)
                atom_clos = ['C', 'O', 'N', 'Other']
                temp_df = pd.DataFrame(np.zeros((len_df, len(atom_clos))), columns=atom_clos)
                self.df = pd.concat([self.df, temp_df], axis=1)
                self.df.loc[self.df.atom_name == 'C', 'C'] = 1
                self.df.loc[self.df.atom_name == 'O', 'O'] = 1
                self.df.loc[self.df.atom_name == 'N', 'N'] = 1
                self.df.loc[(self.df.atom_name != 'C') & (self.df.atom_name != 'O') & (self.df.atom_name != 'N'), 'Other'] = 1
                # residue level (C H O N S)
                res_cols = ['res_C', 'res_H', 'res_O', 'res_N', 'res_Other']
                temp_df = pd.DataFrame(np.zeros((len_df, len(res_cols))), columns=res_cols)
                self.df = pd.concat([self.df, temp_df], axis=1)
                res_set = set(self.df.loc[:,'res'].values)
                for res3letter in res_set:
                    self.df.loc[self.df.res == res3letter, res_cols] = self.aa_vec_dict[aa_321dict[res3letter]]

            # ----------------------------------------------------------------------------------------------------------
            ## pharm encoding vector
            if feature == 'pharm':
                # atom level
                atom_class = ['hydrophobic', 'positive', 'negative', 'neutral', 'acceptor', 'donor', 'aromatic', 'sulphur']
                pharmlst = []
                for i in range(len_df):
                    atom_chain, atom_res, atom_het, atom_posid, atom_inode, atom_full_name, atom_name = self.df.iloc[i,:].values[0:7]
                    aa_pharm_dict_tmp = self.aa_atom_pharm_dict[atom_res]
                    try:
                        pharmlst.append(aa_pharm_dict_tmp[atom_full_name])
                    except:
                        print('[WARNING] atom %-5s do not assigned by pharm, pdbid: %s, atom_chain: %s, atom_pos: %-6s, atom_res: %s'
                              %(atom_full_name, pdbid, atom_chain, str(atom_posid)+str(atom_inode), atom_res))
                        pharmlst.append(aa_pharm_dict_tmp[atom_full_name[0]])  # for unassigned atom in xscore, such as OXT.
                pharm_df = pd.DataFrame(np.array(pharmlst).reshape(len_df, len(atom_class)), columns=atom_class)
                self.df = pd.concat([self.df, pharm_df], axis=1)
                # residue level
                res_cols = ['res_hydrophobic', 'res_positive', 'res_negative', 'res_neutral', 'res_acceptor', 'res_donor',
                            'res_aromatic', 'res_sulphur']
                temp_df = pd.DataFrame(np.zeros((len_df, len(res_cols))), columns=res_cols)
                self.df = pd.concat([self.df, temp_df], axis=1)
                res_set = set(self.df.loc[:, 'res'].values)
                for res3letter in res_set:
                    self.df.loc[self.df.res == res3letter, res_cols] = self.aa_pharm_dict[aa_321dict[res3letter]]
            # ----------------------------------------------------------------------------------------------------------
            ## HP encoding vector.
            if feature == 'hp':
                # atom level
                atom_class = ['hydrophobic_bak', 'polar']
                hplst = []
                for i in range(len_df):
                    atom_chain, atom_res, atom_het, atom_posid, atom_inode, atom_full_name, atom_name = self.df.iloc[i, :].values[0:7]
                    aa_hp_dict_tmp = self.aa_atom_hp_dict[atom_res.upper()]
                    try:
                        hplst.append(aa_hp_dict_tmp[atom_full_name])
                    except:
                        # print('[WARNING] atom %-5s do not assigned by hp, pdbid: %s, atom_chain: %s, atom_pos: %-6s, atom_res: %s'
                        #       %(atom_full_name, pdbid, atom_chain, str(atom_posid)+str(atom_inode), atom_res))
                        hplst.append([0, 0])  # for unassigned atom in xscore at HP classification.
                hp_df = pd.DataFrame(np.array(hplst).reshape(len_df, len(atom_class)), columns=atom_class)
                self.df = pd.concat([self.df, hp_df], axis=1)
                # residue level
                res_cols = ['res_hydrophobic_bak', 'res_polar']
                temp_df = pd.DataFrame(np.zeros((len_df, len(res_cols))), columns=res_cols)
                self.df = pd.concat([self.df, temp_df], axis=1)
                res_set = set(self.df.loc[:, 'res'].values)
                for res3letter in res_set:
                    self.df.loc[self.df.res == res3letter, res_cols] = self.aa_hp_dict[aa_321dict[res3letter]]
            # ----------------------------------------------------------------------------------------------------------
            ## mass value, atom oriented.
            if feature == 'mass':
                temp_df = pd.DataFrame(np.zeros((len_df, 4)), columns=['C_mass','O_mass','N_mass','S_mass'])
                self.df = pd.concat([self.df, temp_df], axis=1)
                self.df.loc[self.df.atom_name == 'C', 'C_mass'] = self.aa_atom_mass['C']
                self.df.loc[self.df.atom_name == 'O', 'O_mass'] = self.aa_atom_mass['O']
                self.df.loc[self.df.atom_name == 'N', 'N_mass'] = self.aa_atom_mass['N']
                self.df.loc[self.df.atom_name == 'S', 'S_mass'] = self.aa_atom_mass['S']
                # return df
            # ----------------------------------------------------------------------------------------------------------
            ## "mutation descriptor" ---> delta residue based on the change about residue_onehot_encoding of both wild and mutant residues.
            if feature == 'deltar':
                delta_r = np.array(self.aa_vec_dict[mtaa]) - np.array(self.aa_vec_dict[wtaa])
                if self.reverse:
                    delta_r = -delta_r
                temp_df = pd.DataFrame(np.ones((len_df, 1)))
                self.df['dC'] = temp_df * delta_r[0]
                self.df['dH'] = temp_df * delta_r[1]
                self.df['dO'] = temp_df * delta_r[2]
                self.df['dN'] = temp_df * delta_r[3]
                self.df['dOther'] = temp_df * delta_r[4]
                # return df
            # ----------------------------------------------------------------------------------------------------------
            ## "mutation descriptor" ---> delta residue based on the change about residue_pharm_encoding of both wild and mutant residues.
            if feature == 'pharm_deltar':
                pharm_delta_r = np.array(self.aa_pharm_dict[mtaa]) - np.array(self.aa_pharm_dict[wtaa])
                if self.reverse:
                    pharm_delta_r = -pharm_delta_r
                temp_df = pd.DataFrame(np.ones((len_df, 1)))
                self.df['dhydrophobic'] = temp_df * pharm_delta_r[0]
                self.df['dpositive'] = temp_df * pharm_delta_r[1]
                self.df['dnegative'] = temp_df * pharm_delta_r[2]
                self.df['dneutral'] = temp_df * pharm_delta_r[3]
                self.df['dacceptor'] = temp_df * pharm_delta_r[4]
                self.df['ddonor'] = temp_df * pharm_delta_r[5]
                self.df['daromatic'] = temp_df * pharm_delta_r[6]
                self.df['dsulphur'] = temp_df * pharm_delta_r[7]
                # return df
            # ----------------------------------------------------------------------------------------------------------
            ## "mutation descriptor" ---> delta residue based on the change about residue_HP_encoding of both wild and mutant residues.
            if feature == 'hp_deltar':
                hp_delta_r = np.array(self.aa_hp_dict[mtaa]) - np.array(self.aa_hp_dict[wtaa])
                if self.reverse:
                    hp_delta_r = -hp_delta_r
                temp_df = pd.DataFrame(np.ones((len_df, 1)))
                self.df['dhydrophobic_bak'] = temp_df * hp_delta_r[0]
                self.df['dpolar'] = temp_df * hp_delta_r[1]
                # return df
            # ----------------------------------------------------------------------------------------------------------
            ## msa frq vector, entropy and delta entropy, residue oriented.
            if feature == 'msa':
                entWTlst = []
                entMTlst = []
                entropylst = []
                WTmsalst = []
                MTmsalst = []

                for i in range(len_df):
                    # for each atom, calc msa features.
                    atom_chain, atom_res, atom_het, atom_posid, atom_inode, atom_full_name, atom_name = self.df.iloc[i, :].values[0:7]
                    atom_position = (str(atom_het) + str(atom_posid) + str(atom_inode)).strip()
                    # print('\nwtblastdir:%s\nmtblastdir:%s'%(self.wtblastdir,self.mtblastdir))
                    if not self.reverse:
                        # calc for wild structure when reverse is False.
                        wtblastdir = '%s/%s/msa.cnt_frq.npz'%(self.wtblastdir,atom_chain)
                        rvWT, entWT = self.calEntropy(wtblastdir, atom_position)
                        if atom_chain == mtchain:
                            rvMT, entMT = self.calEntropy(self.mtblastdir, atom_position)
                        else:
                            rvMT, entMT = rvWT, entWT
                    else:
                        # calc for mutant structure when reverse is True.
                        mtblastdir = '%s/%s/msa.cnt_frq.npz'%(self.wtblastdir,atom_chain)
                        rvMT, entMT = self.calEntropy(mtblastdir, atom_position)
                        if atom_chain == mtchain:
                            rvWT, entWT = self.calEntropy(self.mtblastdir, atom_position)
                        else:
                            rvWT, entWT = rvMT, entMT

                    ## ----------------------------This block for debugging only!---------------------------------------
                    # try:
                    #     assert rvMT.shape[0] ==21# @@++
                    # except:
                    #     print('ERROR at rvMT',self.reverse)
                    #     if self.reverse:
                    #         print(self.mtblastdir, atom_position)
                    #     else:
                    #         print(self.mtblastdir,atom_position)
                    #
                    # try:
                    #     assert rvWT.shape[0] ==21# @@++
                    # except:
                    #     print('ERROR at rvWT',self.reverse)
                    #     if self.reverse:
                    #         print(wtblastdir, atom_position)
                    #     else:
                    #         print(mtblastdir,atom_position)
                    ## -------------------------------------------------------------------------------------------------

                    ent = entMT - entWT

                    entWTlst.append(entWT)
                    entMTlst.append(entMT)
                    entropylst.append(ent)
                    WTmsalst.append(rvWT)
                    MTmsalst.append(rvMT)

                self.df['dEntropy'] = pd.DataFrame(np.array(entropylst).reshape(len_df, 1))
                self.df['entWT'] = pd.DataFrame(np.array(entWTlst).reshape(len_df, 1))
                self.df['entMT'] = pd.DataFrame(np.array(entMTlst).reshape(len_df, 1))
                cols = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', '-']
                MTcols = ['MT_' + aa for aa in cols]
                WTcols = ['WT_' + aa for aa in cols]
                WTmsadf = pd.DataFrame(np.array(WTmsalst).reshape(len_df, 21), columns=WTcols)
                MTmsadf = pd.DataFrame(np.array(MTmsalst).reshape(len_df, 21), columns=MTcols)
                self.df = pd.concat([self.df, WTmsadf, MTmsadf], axis=1)
            # ----------------------------------------------------------------------------------------------------------
            ## rosetta energy, residue oriented.
            if feature == 'energy':
                temp_df = pd.DataFrame(np.zeros((len_df, len(self.energy_name_lst))),columns=self.energy_name_lst)
                df_map    = read_csv(self.mappingdir)
                df_map[['CHAIN']] = df_map[['CHAIN']].astype(str)
                df_map[['POSITION_OLD']] = df_map[['POSITION_OLD']].astype(str)

                df_energy = read_csv(self.energydir)
                df_energy.insert(0, 'res', df_energy.iloc[:,0])
                for i in range(len(df_energy)):
                    ## add one column for indexing
                    df_energy.iloc[i,0] = df_energy.iloc[i,0][:3]
                    df_energy.iloc[i,1] = df_energy.iloc[i, 1].split('_')[-1]

                for i in range(len_df):
                    atom_chain, atom_res, het, posid, inode = self.df.iloc[i,:5]
                    ## ----------------------------This block for debugging only!---------------------------------------
                    # print('atom_chain: %s, atom_res: %s, het: %s, posid: %s, inode: %s' % (
                    # atom_chain, atom_res, het, posid, inode))
                    # # print(type(atom_chain))# str
                    # print(self.mappingdir)
                    ## -------------------------------------------------------------------------------------------------
                    map_new = df_map.loc[(df_map.CHAIN==atom_chain) & (df_map.POSITION_OLD==(str(het)+str(posid)+str(inode)).strip()),:].values[0,-1]

                    energy = df_energy.loc[(df_energy.res==atom_res) & (df_energy.label==str(map_new)),self.energy_name_lst]
                    temp_df.iloc[i,:] = energy.values.reshape(1,-1)

                self.df = pd.concat([self.df, temp_df], axis=1)
            # ----------------------------------------------------------------------------------------------------------
            ## ddg, mutation oriented.
            if feature == 'ddg':
                temp_df = pd.DataFrame(np.ones((len_df, 1)) * self.ddg)
                if self.reverse:
                    temp_df = pd.DataFrame(np.ones((len_df, 1)) * (-self.ddg))
                self.df['ddg'] = temp_df
                # return df
        return self.df

if __name__ == '__main__':
    main()