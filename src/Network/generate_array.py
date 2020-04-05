#!/usr/bin/env python
import os,sys
import numpy as np
import pandas as pd
from mCNN.processing import shell, transform, log

'''
transform feature_csv to numpy array (based on mutation csv which appoint each mut_tag to a fold)
S2648.csv_cross_valid_position_level.fold1.test.csv
      S2648.csv_blind_position_level.test.csv
'''
def main():
    main_pro_pos(rm=True)

def main_pro_pos(rm=False):
    dataset_name = sys.argv[1]
    if rm:
        os.system('rm -rf /public/home/sry/mCNN/dataset/%s/npz/wild/*'%dataset_name)
        os.system('rm -rf /public/home/sry/mCNN/dataset/%s/npz/mutant/*'%dataset_name)
    homedir = shell('echo $HOME')
    k_neighborlst = [30, 40, 50, 60, 70,80,90,100,110,120,130,140,150,160,170,180,190,200]
    # k_neighborlst = [50, 120]
    centerlst = ['CA']
    pca = False
    filelst = os.listdir('/public/home/sry/mCNN/dataset/%s/cross_validation'%dataset_name)

    ## for wild
    feature_csv_dir = '%s/mCNN/dataset/SSD/feature/mCNN/wild/csv' % homedir
    outdir = '%s/mCNN/dataset/%s/npz/wild/cross_valid' % (homedir, dataset_name)
    for file in filelst:
        mutant_csv_pth='/public/home/sry/mCNN/dataset/%s/cross_validation/'%dataset_name+file
        filename_prefix = file.split('_')[-2][:3]+'_'+file.split('.')[-3]+'_'+file.split('.')[-2]
        AG = ArrayGenerator(homedir, dataset_name, mutant_csv_pth, feature_csv_dir, outdir, filename_prefix, k_neighborlst, centerlst, pca=pca)
        AG.array_runner()

    os.chdir('/public/home/sry/mCNN/dataset/%s/npz/wild')
    os.system('mv ./err.lst ./err_dup.lst')
    os.system('sort -u ./err_dup.lst > ./err.lst')

    ## for mutant
    feature_csv_dir = '%s/mCNN/dataset/SSD/feature/mCNN/mutant/csv' % homedir
    outdir = '%s/mCNN/dataset/%s/npz/mutant/cross_valid' % (homedir, dataset_name)
    for file in filelst:
        mutant_csv_pth = '/public/home/sry/mCNN/dataset/%s/cross_validation/' % dataset_name + file
        filename_prefix = file.split('_')[-2][:3] + '_' + file.split('.')[-3] + '_' + file.split('.')[-2]
        AG = ArrayGenerator(homedir, dataset_name, mutant_csv_pth, feature_csv_dir, outdir, filename_prefix,
                            k_neighborlst, centerlst, pca=pca)
        AG.array_runner()

    os.chdir('/public/home/sry/mCNN/dataset/%s/npz/mutant')
    os.system('mv ./err.lst ./err_dup.lst')
    os.system('sort -u ./err_dup.lst > ./err.lst')


def blind_test():
    pass

class ArrayGenerator(object):
    def __init__(self, homedir, dataset_name, mutant_csv_pth, feature_csv_dir, outdir, filename_prefix, k_neighborlst, centerlst, pca=False):
        self.homedir         = homedir
        self.dataset_name    = dataset_name
        self.mutant_csv_pth  = mutant_csv_pth #~/mCNN/dataset/S2648/cross_validation/S2648.csv_cross_valid_position_level.fold4.test.csv
        self.feature_csv_dir = feature_csv_dir # .../feature/mCNN/[wild|mutant]/csv
        self.outdir          = outdir # dataset/S2648/npz/wild
        self.filename_prefix = filename_prefix #output filename_prefix
        self.k_neighborlst   = k_neighborlst
        self.centerlst       = centerlst
        self.pca             = pca

        # set output dir for array
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)

        self.keys = ['dist', 'x', 'y', 'z', 'occupancy', 'b_factor',

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
    @log
    def array_runner(self):
        df_mut = pd.read_csv(self.mutant_csv_pth) #S2648.csv_cross_valid_position_level.fold4.test.csv
        for neighbor in self.k_neighborlst:
            for center in self.centerlst:
                filename_suffix = 'center_%s_PCA_%s_neighbor_%s' % (center, self.pca, neighbor)
                filename = self.filename_prefix+'_'+filename_suffix
                csvpthlst = []
                phlst = []
                thermlst = []
                center_coordpthlst = []
                for i in range(len(df_mut)):
                    mut_tag = '_'.join([str(x) for x in df_mut.iloc[i, :][['PDB','WILD_TYPE','CHAIN','POSITION','MUTANT']].values])
                    wild_feature_csv_pth = '%s/%s/center_%s_neighbor_%s.csv'%(self.feature_csv_dir,mut_tag,center,neighbor) #center_CA_neighbor_30.csv
                    center_coord_pth = '%s/%s/center_%s_neighbor_._center_coord.npy'%(self.feature_csv_dir, mut_tag, center)
                    if os.path.exists(wild_feature_csv_pth) and os.path.exists(center_coord_pth):
                        ph,temperature = df_mut.iloc[i, :][['PH','TEMPERATURE']].values
                        csvpthlst.append(wild_feature_csv_pth)
                        phlst.append(ph)
                        thermlst.append(temperature)
                        center_coordpthlst.append(center_coord_pth)
                    else:
                        with open('%s/err.lst' % self.outdir, mode='a') as f:
                            f.writelines('%s\n' % mut_tag)

                self.array_generator(filename=filename, csvpthlst=csvpthlst, phlst=phlst, thermlst=thermlst,
                                         outdir=self.outdir, center_coordpthlst=center_coordpthlst)


    def array_generator(self, filename, csvpthlst,phlst,thermlst,outdir,center_coordpthlst=None):
        """
        integrating csv files in the csvpthlst to a npz array file and saving it to outdir.
        :param filename: str, the output npz file name.
         e.g. "all"_center_CA_PCA_False_neighbor_30.npz              #integrate all items in the dataset
              "pro_fold1_train"_center_CA_PCA_False_neighbor_30.npz  #for those dataset that tested with protein level
              "pos_fold2_test"_center_CA_PCA_False_neighbor_30.npz   #for those dataset that tested with protein level
              "fold1_train"_center_CA_PCA_False_neighbor_30.npz      #for those dataset that were splited randomly
              "fold2_test"_center_CA_PCA_False_neighbor_30.npz       #for those dataset that were splited randomly
        :param csvdirlst: python list, feature-csv-file paths of all items in this subject (such as all csv file pth of "fold_1_train")
        :param phlst: python lst, PH value list with elements that one-to-one correspondent with each item in the csvpthlst
        :param thermlst: python lst, TEMPERATURE value list with elements that one-to-one correspondent with each item in the csvpthlst
        :param center_coordpthlst: python list, center_coord file path list with elements that one-to-one correspondent with each item in the csvpthlst.
                                   Used when pca is True
        :return: None
        """
        # print(csvpthlst,phlst,thermlst)
        ddglst = []
        ylst = []
        arrlst = []
        for i in range(len(csvpthlst)):
            csvpth      = csvpthlst[i]
            df = pd.read_csv(csvpth)
            df.loc[:, 'ph'] = phlst[i]
            df.loc[:, 'temperature'] = thermlst[i]
            ddg = df.loc[:, 'ddg'].values[0]
            ddglst.append(ddg)
            if ddg >= 0:
                ylst.append(1)
            else:
                ylst.append(0)
            tmp_arr = df.loc[:, self.keys].values
            if self.pca:
                assert center_coordpthlst is not None
                center_coord = np.load(center_coordpthlst[i])
                tmp_arr[:, 1:4] = transform(tmp_arr[:, 1:4], center_coord)

            arrlst.append(tmp_arr)

        x = np.array(arrlst).reshape(len(csvpthlst), -1, len(self.keys))
        ddg = np.array(ddglst).reshape(-1, 1)
        y = np.array(ylst).reshape(-1, 1)
        print('x.shape',x.shape,'y.shape',y.shape,'ddg.shape',ddg.shape)
        assert x.shape[0] == ddg.shape[0] and ddg.shape[0] == y.shape[0]

        if not os.path.exists(outdir):
            os.system('mkdir -p %s' % outdir)
        np.savez('%s/%s.npz' % (outdir, filename), x=x, y=y, ddg=ddg)
        print('npz array stores at [%s/%s.npz]' % (outdir, filename))

if __name__ == '__main__':
    main()