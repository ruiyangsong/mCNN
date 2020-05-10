#!/usr/bin/env python
import os,sys
import numpy as np
import pandas as pd
from mCNN.processing import transform, log

'''
transform feature_csv to numpy array (based on fold-specific mutation csv which appoint each mutation to a fold)
the format of fold-specified csv file are "{dataset_name.csv}_{some_markers}.{fold_num}.{train_or_test}.{csv}"
example of fold-specific csv files:
{S2648.csv}_{cross_valid_position_level}.{fold1}.{test}.csv]
 or 
{deepddg.csv}_{cross_valid}.{fold1}.{train}.csv]
'''
def main():
    if len(sys.argv) == 1:
        print('Usage: generate_array.py [dataset_name]')
        sys.exit(0)

    dataset_name  = sys.argv[1]
    kneighbor_lst = [120]
    radii_lst     = [12]
    center_lst    = ['CA']
    atom_lst      = None
    pca           = False


    ## wild
    # wild_feature_csv_dir = '/public/home/sry/mCNN/dataset/' + dataset_name + '/feature/mCNN/wild/csv'
    wild_feature_csv_dir = '/public/home/sry/mCNN/dataset/SSD/feature/mCNN/wild/csv'
    wild_outdir = '/public/home/sry/mCNN/dataset/' + dataset_name + '/npz/wild/cross_valid'
    run_neighbor(dataset_name, wild_feature_csv_dir, wild_outdir, atom_lst, kneighbor_lst, center_lst, pca)
    run_radii(dataset_name, wild_feature_csv_dir, wild_outdir, atom_lst, radii_lst, center_lst, pca)

    ## mutant
    # mutant_feature_csv_dir = '/public/home/sry/mCNN/dataset/' + dataset_name + '/feature/mCNN/mutant/csv'
    mutant_feature_csv_dir = '/public/home/sry/mCNN/dataset/SSD/feature/mCNN/mutant/csv'
    mutant_outdir = '/public/home/sry/mCNN/dataset/' + dataset_name + '/npz/mutant/cross_valid'
    run_neighbor(dataset_name, mutant_feature_csv_dir, mutant_outdir, atom_lst, kneighbor_lst, center_lst, pca)
    run_radii(dataset_name, mutant_feature_csv_dir, mutant_outdir, atom_lst, radii_lst, center_lst, pca)


def run_neighbor(dataset_name,feature_csv_dir,outdir,atom_lst,kneighbor_lst,center_lst,pca):
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    fold_csv_dir = '/public/home/sry/mCNN/dataset/' + dataset_name + '/cross_validation'
    fold_csv_name_lst = os.listdir(fold_csv_dir)
    for fold_csv_name in fold_csv_name_lst:
        for kneighbor in kneighbor_lst:
            for center in center_lst:
                fold_csv_pth = '/public/home/sry/mCNN/dataset/' + dataset_name + '/cross_validation/' + fold_csv_name
                filename_prefix = fold_csv_name.split('_')[-2][:3] + '_' + fold_csv_name.split('.')[-3] + '_' + fold_csv_name.split('.')[-2]
                filename = filename_prefix + 'center_%s_PCA_%s_neighbor_%s' % (center, pca, kneighbor)
                AG = ArrayGenerator(fold_csv_pth, feature_csv_dir, outdir, filename, atom_lst=atom_lst, center=center, pca=pca)
                AG.neighbor_array_generator(k_neighbor=kneighbor)
    try:
        os.chdir('%s../' % outdir)
        os.system('mv err.lst err_dup.lst && sort -u err_dup.lst > err.lst')
    except:
        pass


def run_radii(dataset_name,feature_csv_dir,outdir,atom_lst,radii_lst,center_lst,pca):
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    fold_csv_dir = '/public/home/sry/mCNN/dataset/' + dataset_name + '/cross_validation'
    fold_csv_name_lst = os.listdir(fold_csv_dir)
    for fold_csv_name in fold_csv_name_lst:
        for radii in radii_lst:
            for center in center_lst:
                fold_csv_pth = '/public/home/sry/mCNN/dataset/' + dataset_name + '/cross_validation/' + fold_csv_name
                filename_prefix = fold_csv_name.split('_')[-2][:3] + '_' + fold_csv_name.split('.')[-3] + '_' + fold_csv_name.split('.')[-2]
                filename = filename_prefix + 'center_%s_PCA_%s_radii_%s' % (center, pca, radii)
                AG = ArrayGenerator(fold_csv_pth, feature_csv_dir, outdir, filename, atom_lst=atom_lst, center=center, pca=pca)
                AG.radii_array_generator(radii=radii)
    try:
        os.chdir('%s../' % outdir)
        os.system('mv err.lst err_dup.lst && sort -u err_dup.lst > err.lst')
    except:
        pass



def blind_test():
    pass

class ArrayGenerator(object):
    def __init__(self, mutant_csv_pth, feature_csv_dir, outdir, filename, atom_lst=None, center='CA', pca=False):
        '''
        Regulate mutations (pre-calculated feature csv for each) listed in mutant_csv_pth to npz file
        :param mutant_csv_pth:
        :param feature_csv_dir: directory which stored feature csv files for each mutation
                                e.g.:
                                [/public/home/sry/mCNN/dataset/S2648/feature/mCNN/mutant/csv]
                                [/public/home/sry/mCNN/dataset/S2648/feature/mCNN/wild/csv]
        :param outdir: output directory
        :param filename: output npz file name
        :param atom_lst: only keep the atoms in atom_lst
        :param center:
        :param pca:
        '''
        self.mutant_csv_pth  = mutant_csv_pth
        self.feature_csv_dir = feature_csv_dir
        self.outdir          = outdir
        self.filename        = filename
        self.atom_lst        = atom_lst
        self.center          = center
        self.pca             = pca

        self._pre_calculations()

    def _pre_calculations(self):
        self.feature_csv_pth_lst = []
        self.center_coord_pth_lst = []
        self.errlst_pth = '%s/../err.lst' % self.outdir
        self.dataset_idx_pth = '%s/../npz.idx' % self.outdir
        self.keys_old = ['dist', 'x', 'y', 'z', 'occupancy', 'b_factor',

                     's_H', 's_G', 's_I', 's_E', 's_B', 's_T', 's_C',
                     's_Helix', 's_Strand', 's_Coil',

                     'sa', 'rsa', 'asa', 'phi', 'psi',

                     'ph', 'temperature',

                     'C', 'O', 'N', 'Other',

                     'C_mass', 'O_mass', 'N_mass', 'S_mass',

                     'hydrophobic', 'positive', 'negative', 'neutral', 'acceptor', 'donor', 'aromatic', 'sulphur',
                     'hydrophobic_bak', 'polar',

                     'fa_atr', 'fa_rep', 'fa_sol', 'fa_intra_rep', 'fa_intra_sol_xover4', 'lk_ball_wtd', 'fa_elec', 'pro_close',
                     'hbond_bb_sc', 'hbond_sc', 'omega', 'fa_dun', 'p_aa_pp', 'yhh_planarity', 'ref', 'rama_prepro', 'total',

                     'WT_A', 'WT_R', 'WT_N', 'WT_D', 'WT_C', 'WT_Q', 'WT_E', 'WT_G', 'WT_H', 'WT_I', 'WT_L', 'WT_K', 'WT_M',
                     'WT_F', 'WT_P', 'WT_S', 'WT_T', 'WT_W', 'WT_Y', 'WT_V', 'WT_-',
                     'MT_A', 'MT_R', 'MT_N', 'MT_D', 'MT_C', 'MT_Q', 'MT_E', 'MT_G', 'MT_H', 'MT_I', 'MT_L', 'MT_K', 'MT_M',
                     'MT_F', 'MT_P', 'MT_S', 'MT_T', 'MT_W', 'MT_Y', 'MT_V', 'MT_-',

                     'dC', 'dH', 'dO', 'dN', 'dOther',

                     'dhydrophobic', 'dpositive', 'dnegative', 'dneutral', 'dacceptor', 'ddonor', 'daromatic', 'dsulphur',

                     'dhydrophobic_bak', 'dpolar',

                     'dEntropy', 'entWT', 'entMT']

        self.keys = ['dist', 'omega_Orient', 'theta12', 'theta21', 'phi12', 'phi21', 'sin_omega', 'cos_omega','sin_theta12',
                     'cos_theta12', 'sin_theta21', 'cos_theta21', 'sin_phi12', 'cos_phi12', 'sin_phi21', 'cos_phi21',
                     'x', 'y', 'z', 'x2CA', 'y2CA', 'z2CA', 'hse_up', 'hse_down', 'occupancy', 'b_factor', 'depth',

                     's_H', 's_G', 's_I', 's_E', 's_B', 's_T', 's_C',
                     's_Helix', 's_Strand', 's_Coil',

                     'sa', 'rsa', 'asa', 'phi', 'psi', 'sin_phi', 'cos_phi', 'sin_psi', 'cos_psi',

                     'ph', 'temperature',

                     'C', 'O', 'N', 'Other',
                     'res_C', 'res_H', 'res_O', 'res_N', 'res_Other',

                     'C_mass', 'O_mass', 'N_mass', 'S_mass',

                     'hydrophobic', 'positive', 'negative', 'neutral', 'acceptor', 'donor', 'aromatic', 'sulphur',
                     'res_hydrophobic', 'res_positive', 'res_negative', 'res_neutral', 'res_acceptor', 'res_donor', 'res_aromatic', 'res_sulphur',

                     'hydrophobic_bak', 'polar',
                     'res_hydrophobic_bak', 'res_polar',

                     'fa_atr', 'fa_rep', 'fa_sol', 'fa_intra_rep', 'fa_intra_sol_xover4', 'lk_ball_wtd', 'fa_elec', 'pro_close',
                     'hbond_bb_sc', 'hbond_sc', 'omega', 'fa_dun', 'p_aa_pp', 'yhh_planarity', 'ref', 'rama_prepro', 'total',

                     'WT_A', 'WT_R', 'WT_N', 'WT_D', 'WT_C', 'WT_Q', 'WT_E', 'WT_G', 'WT_H', 'WT_I', 'WT_L', 'WT_K', 'WT_M',
                     'WT_F', 'WT_P', 'WT_S', 'WT_T', 'WT_W', 'WT_Y', 'WT_V', 'WT_-',
                     'MT_A', 'MT_R', 'MT_N', 'MT_D', 'MT_C', 'MT_Q', 'MT_E', 'MT_G', 'MT_H', 'MT_I', 'MT_L', 'MT_K', 'MT_M',
                     'MT_F', 'MT_P', 'MT_S', 'MT_T', 'MT_W', 'MT_Y', 'MT_V', 'MT_-',

                     'dC', 'dH', 'dO', 'dN', 'dOther',

                     'dhydrophobic', 'dpositive', 'dnegative', 'dneutral', 'dacceptor', 'ddonor', 'daromatic', 'dsulphur',

                     'dhydrophobic_bak', 'dpolar',

                     'dEntropy', 'entWT', 'entMT']
        # set output dir for array
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)
        idx = open(self.dataset_idx_pth,'w')
        df_mutation = pd.read_csv(self.mutant_csv_pth)
        for i in range(len(df_mutation)):
            mut_tag = '_'.join([str(x) for x in df_mutation.iloc[i, :][['PDB','WILD_TYPE','CHAIN','POSITION','MUTANT']].values])
            feature_csv_pth  = '%s/%s/center_%s.csv'%(self.feature_csv_dir,mut_tag,self.center)
            center_coord_pth = '%s/%s/center_%s_neighbor_._center_coord.npy'%(self.feature_csv_dir, mut_tag, self.center)
            if os.path.exists(feature_csv_pth) and os.path.exists(center_coord_pth):
                idx.writelines('%s\n'%mut_tag)
                self.feature_csv_pth_lst.append(feature_csv_pth)
                self.center_coord_pth_lst.append(center_coord_pth)
            else:
                with open(self.errlst_pth, mode='a') as f:
                    f.writelines('%s\n' % mut_tag)
        idx.close()


    def atom_filter(self, df, atom_lst):
        return df.loc[df.full_name.isin(atom_lst), :]

    def calc_neighbor(self, df, kneighbor):
        dist_arr = df.loc[:, 'dist'].values
        assert len(dist_arr) >= kneighbor
        indices = sorted(dist_arr.argsort()[:kneighbor])
        df = df.iloc[indices, :]
        return df

    def calc_radius(self, df, radius=12):
        '''also 12 armstrong for calc HSE'''
        return df.loc[df.dist <= radius, :]

    @log
    def neighbor_array_generator(self,k_neighbor):
        ddglst = []
        ylst   = []
        arrlst = []
        for i in range(len(self.feature_csv_pth_lst)):
            feature_csv_pth = self.feature_csv_pth_lst[i]
            df_feature = pd.read_csv(feature_csv_pth)

            ## filter and calc neighbor
            if self.atom_lst is not None:
                df_feature = self.atom_filter(df=df_feature,atom_lst=self.atom_lst)

            df_feature = self.calc_neighbor(df=df_feature,kneighbor=k_neighbor)

            ## calc ddg and y
            ddg = df_feature.loc[:, 'ddg'].values[0]
            ddglst.append(ddg)
            if ddg >= 0:
                ylst.append(1)
            else:
                ylst.append(0)

            if self.pca:
                center_coord = np.load(self.center_coord_pth_lst[i])
                df_feature.loc[:, ['x','y','z']] = transform(df_feature.loc[:, ['x','y','z']].values, center_coord)

            tmp_arr = df_feature.loc[:, self.keys].values
            arrlst.append(tmp_arr)

        x   = np.array(arrlst).reshape(len(self.feature_csv_pth_lst), -1, len(self.keys))
        ddg = np.array(ddglst).reshape(-1, 1)
        y   = np.array(ylst).reshape(-1, 1)

        np.savez('%s/%s.npz' % (self.outdir, self.filename), x=x, y=y, ddg=ddg)
        print('npz array stores at [%s/%s.npz]' % (self.outdir, self.filename))

    @log
    def radii_array_generator(self,radii):
        ddglst = []
        ylst   = []
        arrlst = []
        for i in range(len(self.feature_csv_pth_lst)):
            feature_csv_pth = self.feature_csv_pth_lst[i]
            df_feature = pd.read_csv(feature_csv_pth)

            ## filter and calc neighbor or radius
            if self.atom_lst is not None:
                df_feature = self.atom_filter(df=df_feature,atom_lst=self.atom_lst)

            df_feature = self.calc_radius(df=df_feature,radius=radii)

            ## calc ddg and y
            ddg = df_feature.loc[:, 'ddg'].values[0]
            ddglst.append(ddg)
            if ddg >= 0:
                ylst.append(1)
            else:
                ylst.append(0)

            if self.pca:
                center_coord = np.load(self.center_coord_pth_lst[i])
                df_feature.loc[:, ['x','y','z']] = transform(df_feature.loc[:, ['x','y','z']].values, center_coord)

            tmp_arr = df_feature.loc[:, self.keys].values
            arrlst.append(tmp_arr)
        ################################################################################################################
        ## different length may happen among samples
        ################################################################################################################
        x   = np.array(arrlst)
        ddg = np.array(ddglst).reshape(-1, 1)
        y   = np.array(ylst).reshape(-1, 1)

        np.savez('%s/%s.npz' % (self.outdir, self.filename), x=x, y=y, ddg=ddg)
        print('npz array stores at [%s/%s.npz]' % (self.outdir, self.filename))

if __name__ == '__main__':
    main()