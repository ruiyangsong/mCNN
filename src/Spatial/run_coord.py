#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
根据rosetta生成的pdb_ref 和 pdb_mut计算mCNN特征并保存至csv，
其中不考虑rosetta mut 失败的条目（根据 rosetta mut 的结果反向去 mt_csv 中查找 ddg温度等 特征指标）
'''

import os, sys, time, argparse
import numpy as np
from mCNN.processing import shell, str2bool, read_csv, log, check_qsub, save_data_array, transform

def main():
    homedir    = shell('echo $HOME')
    featurelst = ' '.join(['rsa', 'thermo', 'onehot', 'pharm', 'hp', 'mass', 'deltar', 'pharm_deltar','hp_deltar', 'msa', 'energy', 'ddg'])
    # ------------------------------------------------------------------------------------------------------------------
    ## parse argument
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_name',       type=str, help='dataset name')
    parser.add_argument('--flag', type=str, choices=['first', 'all'], default='all',
                        help='"first" calc df_feature only, "all" calc df_neighbor and df_feature, default is "all".')

    parser.add_argument('-k', '--k_neighbor', type=int, required=True,   nargs='+', help='The k_neighbor list.')
    parser.add_argument('-C', '--center',     type=str, required=True,   nargs='+', choices=['CA','geometric'], help='The MT site center type.')
    parser.add_argument('-T', '--pca',        type=str, default='False', choices=['False', 'True'], help='If consider pca transform, the default is False')
    args = parser.parse_args()
    dataset_name  = args.dataset_name
    flag          = args.flag
    k_neighborlst = args.k_neighbor
    centerlst     = args.center
    pca           = str2bool(args.pca)

    QR = CoordRunner(homedir,dataset_name,flag,k_neighborlst,centerlst,featurelst)
    QR.coord_runner()

    if flag == 'first':
        ## first means only calc df_feature (ALL the ATOMS), do not run the latter part in this main function
        exit(0)
    # ==================================================================================================================
    ## ArrayGenerator was replaced by another script (../Network/generate_array.py)
    # ==================================================================================================================
    # AG = ArrayGenerator(homedir, dataset_name, k_neighborlst,centerlst,pca)
    # AG.array_runner()
    # ==================================================================================================================
#-----------------------------------------------------------------------------------------------------------------------
class CoordRunner(object):
    def __init__(self,homedir,dataset_name,flag,k_neighborlst,centerlst,featurelst):
        self.homedir       = homedir
        self.dataset_name  = dataset_name
        self.flag          = flag
        self.k_neighborlst = k_neighborlst
        self.centerlst     = centerlst
        self.feature       = featurelst

        self.app           = '%s/mCNN/src/Spatial/coord.py' % homedir

        self.sleep_time    = 5

        self.mt_csv_dir  = '%s/mCNN/dataset/%s/%s.csv' % (homedir, dataset_name, dataset_name)  # after drop duplicates, #PDB,WILD_TYPE,CHAIN,POSITION,MUTANT,PH,TEMPERATURE,DDG
        self.map_csv_dir = '%s/mCNN/dataset/%s/feature/rosetta/global_output/mapping' % (homedir, dataset_name)  # after drop duplicates, #PDB,WILD_TYPE,CHAIN,POSITION,MUTANT,PH,TEMPERATURE,DDG

        self.ref_pdb_dir = '%s/mCNN/dataset/%s/feature/rosetta/ref_output' % (homedir, dataset_name)
        self.mut_pdb_dir = '%s/mCNN/dataset/%s/feature/rosetta/mut_output' % (homedir, dataset_name)

        self.stride_dir  = '%s/mCNN/dataset/%s/feature/stride'%(homedir,dataset_name)
        self.msa_dir     = '%s/mCNN/dataset/%s/feature/msa'%(homedir,dataset_name)

        self.wild_csv_outdir   = '%s/mCNN/dataset/%s/feature/mCNN/wild/csv' % (homedir, dataset_name)
        self.mutant_csv_outdir = '%s/mCNN/dataset/%s/feature/mCNN/mutant/csv' % (homedir, dataset_name)

        self.wild_qsublog_outdir   = '%s/mCNN/dataset/%s/feature/mCNN/wild/qsublog' % (homedir, dataset_name)
        self.mutant_qsublog_outdir = '%s/mCNN/dataset/%s/feature/mCNN/mutant/qsublog' % (homedir, dataset_name)

        if not os.path.exists(self.wild_csv_outdir):
            os.makedirs(self.wild_csv_outdir)
        if not os.path.exists(self.mutant_csv_outdir):
            os.makedirs(self.mutant_csv_outdir)
        if not os.path.exists(self.wild_qsublog_outdir):
            os.makedirs(self.wild_qsublog_outdir)
        if not os.path.exists(self.mutant_qsublog_outdir):
            os.makedirs(self.mutant_qsublog_outdir)

    @log
    def coord_runner(self):
        df_mt = read_csv(self.mt_csv_dir)
        df_mt[['POSITION']] = df_mt[['POSITION']].astype(str)
        for rosetta_mut_tag in os.listdir(self.mut_pdb_dir):
            pdbid, wtaa, chain, pos, _, mtaa = rosetta_mut_tag.split('_')
            mutant_tag = '_'.join([pdbid, wtaa, chain, pos, mtaa])
            try:
                ph, T, ddg = df_mt.loc[(df_mt.PDB==pdbid) & (df_mt.WILD_TYPE==wtaa) & (df_mt.CHAIN==chain) &
                                       (df_mt.POSITION==pos) & (df_mt.MUTANT==mtaa),['PH','TEMPERATURE','DDG']].values[0,:]
                thermo = str(ph) + ' ' + str(T)
            except:
                print('[ERROR] mutation %s index failed in %s'%(mutant_tag,self.mt_csv_dir))
                sys.exit(1)

            for k_neighbor in self.k_neighborlst:
                for center in self.centerlst:
                    self.run_cal_wild(mutant_tag,k_neighbor,center,thermo,ddg)
                    self.run_cal_mutant(rosetta_mut_tag,mutant_tag,k_neighbor,center,thermo,ddg)

        check_qsub(tag='coord_%s'%self.dataset_name, sleep_time=self.sleep_time)

    def run_cal_wild(self,mutant_tag,k_neighbor,center,thermo,ddg):
        reverse = 'False'
        pdbid, wtaa, chain, pos, mtaa = mutant_tag.split('_')
        # -----------------------------Extra parameters for NeighborCalculator------------------------------------------
        pdbdir = '%s/%s/%s_ref.pdb' % (self.ref_pdb_dir, pdbid, pdbid)
        # -----------------------------Extra parameters for FeatureGenerator--------------------------------------------
        wt_blast_path = '%s/mdl0_wild/%s'%(self.msa_dir,pdbid)
        mt_blast_dir = '%s/mdl0_mutant/%s/msa.cnt_frq.npz'%(self.msa_dir,mutant_tag)

        energy_dir   = '%s/%s/energy.csv' % (self.ref_pdb_dir, pdbid)
        mapping_dir  = '%s/%s.csv'%(self.map_csv_dir,pdbid)
        sa_dir       = '%s/wild/%s.stride' % (self.stride_dir, pdbid)
        # -----------------------------qsub-----------------------------------------------------------------------------
        filename = 'center_%s_neighbor_%s'%(center,k_neighbor)

        qsubid = 'coord_%s_wild_%s_%s' % (self.dataset_name,mutant_tag,filename)
        csv_outdir = '%s/%s' % (self.wild_csv_outdir, mutant_tag)
        qsublog_outdir = '%s/%s/%s' % (self.wild_qsublog_outdir, mutant_tag,filename)
        if not os.path.exists(csv_outdir):
            os.makedirs(csv_outdir)
        if not os.path.exists(qsublog_outdir):
            os.makedirs(qsublog_outdir)

        walltime = 'walltime=24:00:00'
        errfile = '%s/err' % qsublog_outdir
        outfile = '%s/out' % qsublog_outdir
        run_prog = '%s/run_prog.sh' %qsublog_outdir

        g = open(run_prog, 'w+')
        g.writelines('#!/usr/bin/env bash\n')
        g.writelines("echo 'user:' `whoami`\necho 'hostname:' `hostname`\necho 'begin at:' `date`\n")
        g.writelines(
            '%s --flag %s -p %s -tag %s -k %s -c %s -o %s -n %s --reverse %s -f %s --wtblastdir %s --mtblastdir %s --energydir %s --mappingdir %s -S %s -t %s -d %s\n'
            % (self.app, self.flag, pdbdir, mutant_tag, k_neighbor, center, csv_outdir, filename,reverse, self.feature, wt_blast_path,mt_blast_dir,energy_dir, mapping_dir, sa_dir, thermo, ddg))
        g.writelines("echo 'end at:' `date`\n")
        g.close()
        os.system('chmod 755 %s' % run_prog)
        os.system('%s/bin/getQ.pl'%self.homedir)
        os.popen('qsub -e %s -o %s -l %s -N %s %s' % (errfile, outfile, walltime, qsubid, run_prog))
        time.sleep(0.01)

    def run_cal_mutant(self, rosetta_mut_tag,mutant_tag,k_neighbor,center,thermo,ddg):
        reverse = 'True'
        pdbid, wtaa, chain, pos, mtaa = mutant_tag.split('_')
        # -----------------------------parameters for NeighborCalculator-----------------------------
        pdbdir = '%s/%s/%s_mut.pdb' % (self.mut_pdb_dir, rosetta_mut_tag, pdbid)
        # -----------------------------parameters for FeatureGenerator-----------------------------
        wt_blast_path = '%s/mdl0_wild/%s/' % (self.msa_dir, pdbid)
        mt_blast_dir = '%s/mdl0_mutant/%s/msa.cnt_frq.npz' % (self.msa_dir, mutant_tag)

        energy_dir = '%s/%s/energy.csv' % (self.mut_pdb_dir, rosetta_mut_tag)
        mapping_dir = '%s/%s.csv' % (self.map_csv_dir, pdbid)
        sa_dir = '%s/mutant/%s.stride' % (self.stride_dir, rosetta_mut_tag)
        # -----------------------------qsub-----------------------------
        filename = 'center_%s_neighbor_%s'%(center,k_neighbor)

        qsubid = 'coord_%s_mutant_%s_%s' % (self.dataset_name,mutant_tag,filename)
        csv_outdir = '%s/%s/' % (self.mutant_csv_outdir, mutant_tag)
        qsublog_outdir = '%s/%s/%s' % (self.mutant_qsublog_outdir, mutant_tag, filename)
        if not os.path.exists(csv_outdir):
            os.makedirs(csv_outdir)
        if not os.path.exists(qsublog_outdir):
            os.makedirs(qsublog_outdir)

        walltime = 'walltime=24:00:00'
        errfile = '%s/err' % qsublog_outdir
        outfile = '%s/out' % qsublog_outdir
        run_prog = '%s/run_prog.sh' % qsublog_outdir

        g = open(run_prog, 'w+')
        g.writelines('#!/usr/bin/env bash\n')
        g.writelines("echo 'user:' `whoami`\necho 'hostname:' `hostname`\necho 'begin at:' `date`\n")
        g.writelines(
            '%s --flag %s -p %s -tag %s -k %s -c %s -o %s -n %s --reverse %s -f %s --wtblastdir %s --mtblastdir %s --energydir %s --mappingdir %s -S %s -t %s -d %s\n'
            % (self.app, self.flag, pdbdir, mutant_tag, k_neighbor, center, csv_outdir, filename,reverse, self.feature,
               wt_blast_path, mt_blast_dir, energy_dir, mapping_dir, sa_dir, thermo, ddg))
        g.writelines("echo 'end at:' `date`\n")
        g.close()
        os.system('chmod 755 %s' % run_prog)
        os.system('%s/bin/getQ.pl'%self.homedir)
        os.popen('qsub -e %s -o %s -l %s -N %s %s' % (errfile, outfile, walltime, qsubid, run_prog))
        time.sleep(0.01)



# ======================================================================================================================
# Generating numpy array of the whole dataset. see ../Network/generate_array.py
# ======================================================================================================================

# class ArrayGenerator(object):
#     def __init__(self,homedir, dataset_name, k_neighborlst,centerlst,pca):
#         self.homedir       = homedir
#         self.dataset_name  = dataset_name
#         self.k_neighborlst = k_neighborlst
#         self.centerlst     = centerlst
#         self.pca           = pca
#         self.dataset_dir     = '%s/mCNN/dataset/%s' % (self.homedir,self.dataset_name)
#         self.wild_csv_path   = '%s/feature/mCNN/wild/csv' % (self.dataset_dir)
#         self.mutant_csv_path = '%s/feature/mCNN/mutant/csv' % (self.dataset_dir)
#
#         # set output dir for feature array
#         self.wild_outdir_k   = '%s/mCNN/dataset/%s/feature/mCNN/wild/npz' % (self.homedir, self.dataset_name)
#         self.mutant_outdir_k = '%s/mCNN/dataset/%s/feature/mCNN/mutant/npz' % (self.homedir, self.dataset_name)
#         if not os.path.exists(self.wild_outdir_k):
#             os.makedirs(self.wild_outdir_k)
#         if not os.path.exists(self.mutant_outdir_k):
#             os.makedirs(self.mutant_outdir_k)
#
#         self.keys = ['dist', 'x', 'y', 'z', 'occupancy', 'b_factor',
#
#                      's_H', 's_G', 's_I', 's_E', 's_B', 's_T', 's_C',
#                      's_Helix', 's_Strand', 's_Coil',
#
#                      'sa', 'rsa', 'asa', 'phi', 'psi',
#
#                      'ph', 'temperature',
#
#                      'C', 'O', 'N', 'Other',
#
#                      'C_mass', 'O_mass', 'N_mass', 'S_mass',
#
#                      'hydrophobic', 'positive', 'negative', 'neutral', 'acceptor', 'donor', 'aromatic', 'sulphur',
#                      'hydrophobic_bak', 'polar',
#
#                      'fa_atr', 'fa_rep', 'fa_sol', 'fa_intra_rep', 'fa_intra_sol_xover4', 'lk_ball_wtd', 'fa_elec', 'pro_close',
#                      'hbond_bb_sc', 'hbond_sc', 'omega', 'fa_dun', 'p_aa_pp', 'yhh_planarity', 'ref', 'rama_prepro', 'total',
#
#                      'WT_A', 'WT_R', 'WT_N', 'WT_D', 'WT_C', 'WT_Q', 'WT_E', 'WT_G', 'WT_H', 'WT_I', 'WT_L', 'WT_K', 'WT_M',
#                      'WT_F', 'WT_P', 'WT_S', 'WT_T', 'WT_W', 'WT_Y', 'WT_V', 'WT_-',
#                      'MT_A', 'MT_R', 'MT_N', 'MT_D', 'MT_C', 'MT_Q', 'MT_E', 'MT_G', 'MT_H', 'MT_I', 'MT_L', 'MT_K', 'MT_M',
#                      'MT_F', 'MT_P', 'MT_S', 'MT_T', 'MT_W', 'MT_Y', 'MT_V', 'MT_-',
#
#                      'dC', 'dH', 'dO', 'dN', 'dOther',
#
#                      'dhydrophobic', 'dpositive', 'dnegative', 'dneutral', 'dacceptor', 'ddonor', 'daromatic', 'dsulphur',
#
#                      'dhydrophobic_bak', 'dpolar',
#
#                      'dEntropy', 'entWT', 'entMT']
#
#         # self.keys = ['dist', 'x', 'y', 'z', 'occupancy', 'b_factor',
#         #
#         #              's_H', 's_G', 's_I', 's_E', 's_B', 's_T', 's_C',
#         #              's_Helix', 's_Strand', 's_Coil',
#         #
#         #              'sa', 'rsa', 'asa', 'phi', 'psi',
#         #
#         #              'ph', 'temperature',
#         #
#         #              'C', 'O', 'N', 'Other',
#         #
#         #              'C_mass', 'O_mass', 'N_mass', 'S_mass',
#         #
#         #              'hydrophobic', 'positive', 'negative', 'neutral', 'acceptor', 'donor', 'aromatic', 'sulphur',
#         #              'hydrophobic_bak', 'polar',
#         #
#         #              'fa_atr', 'fa_rep', 'fa_sol', 'fa_intra_rep', 'fa_intra_sol_xover4', 'lk_ball_wtd', 'fa_elec', 'pro_close',
#         #              'hbond_sr_bb', 'hbond_lr_bb', 'hbond_bb_sc', 'hbond_sc', 'dslf_fa13', 'atom_pair_constraint',
#         #              'angle_constraint', 'dihedral_constraint', 'omega', 'fa_dun', 'p_aa_pp', 'yhh_planarity', 'ref',
#         #              'rama_prepro', 'total',
#         #
#         #              'WT_A', 'WT_R', 'WT_N', 'WT_D', 'WT_C', 'WT_Q', 'WT_E', 'WT_G', 'WT_H', 'WT_I', 'WT_L', 'WT_K', 'WT_M',
#         #              'WT_F', 'WT_P', 'WT_S', 'WT_T', 'WT_W', 'WT_Y', 'WT_V', 'WT_-',
#         #              'MT_A', 'MT_R', 'MT_N', 'MT_D', 'MT_C', 'MT_Q', 'MT_E', 'MT_G', 'MT_H', 'MT_I', 'MT_L', 'MT_K', 'MT_M',
#         #              'MT_F', 'MT_P', 'MT_S', 'MT_T', 'MT_W', 'MT_Y', 'MT_V', 'MT_-',
#         #
#         #              'dC', 'dH', 'dO', 'dN', 'dOther',
#         #
#         #              'dhydrophobic', 'dpositive', 'dnegative', 'dneutral', 'dacceptor', 'ddonor', 'daromatic', 'dsulphur',
#         #
#         #              'dhydrophobic_bak', 'dpolar',
#         #
#         #              'dEntropy', 'entWT', 'entMT']
#
#     @log
#     def array_runner(self):
#         wild_tag_lst = os.listdir(self.wild_csv_path)
#         mutant_tag_lst = os.listdir(self.mutant_csv_path)
#         with open('%s/wild_array_idx.lst'%self.dataset_dir,mode='w') as f:
#             for x in wild_tag_lst:
#                 f.writelines('%s\n'%x)
#         with open('%s/mutant_array_idx.lst'%self.dataset_dir,mode='w') as f:
#             for x in mutant_tag_lst:
#                 f.writelines('%s\n'%x)
#
#         for k_neighbor in self.k_neighborlst:
#             for center in self.centerlst:
#                 filename = 'center_%s_PCA_%s_neighbor_%s' % (center, self.pca, k_neighbor)
#                 ## for wild
#                 wild_csvdirlst = [self.wild_csv_path + '/' + x + '/' + 'center_%s_neighbor_%s.csv' % (center, k_neighbor) for x in wild_tag_lst]
#                 self.array_generator(wild_csvdirlst,filename,k_neighbor,center,self.wild_outdir_k)
#
#                 ## for mutant
#                 mutant_csvdirlst = [self.mutant_csv_path + '/' + x + '/' + 'center_%s_neighbor_%s.csv' % (center, k_neighbor) for x in mutant_tag_lst]
#                 self.array_generator(mutant_csvdirlst,filename,k_neighbor,center,self.mutant_outdir_k)
#
#     def array_generator(self, csvdirlst,filename,k_neighbor,center,outdir_k):
#         ddglst = []
#         ylst = []
#         arrlst = []
#         for csvdir in csvdirlst:
#             df = read_csv(csvdir)
#             ddg = df.loc[:, 'ddg'].values[0]
#             ddglst.append(ddg)
#             if ddg >= 0:
#                 ylst.append(1)
#             else:
#                 ylst.append(0)
#             tmp_arr = df.loc[:, self.keys].values
#             if self.pca:
#                 try:
#                     prefix = '/'.join(csvdir.split('/')[:-1])
#                     center_coord_dir = '%s/center_%s_neighbor_%s_center_coord.npy' % (prefix, center, k_neighbor)
#                 except:
#                     center_coord_dir = '%s/center_%s_neighbor_all_center_coord.npy' % (prefix, center)
#                 center_coord = np.load(center_coord_dir)
#                 tmp_arr[:, 1:4] = transform(tmp_arr[:, 1:4], center_coord)
#
#             arrlst.append(tmp_arr)
#
#         x = np.array(arrlst).reshape(-1, k_neighbor, len(self.keys))
#         ddg = np.array(ddglst).reshape(-1, 1)
#         y = np.array(ylst).reshape(-1, 1)
#         assert x.shape[0] == ddg.shape[0] and ddg.shape[0] == y.shape[0]
#         save_data_array(x, y, ddg, filename, outdir_k)

if __name__ == '__main__':
    main()