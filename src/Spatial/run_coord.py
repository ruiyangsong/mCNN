#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
根据rosetta生成的pdb_ref 和 pdb_mut计算mCNN特征并保存至csv，
其中不考虑rosetta mut 失败的条目（根据rosetta结果反向去mt_csv 中查找ddg温度等特征指标）
'''

#'File name format: MT_pdb_wtaa_chain_position_mtaa_serial'

import os, sys, time, argparse
from mCNN.processing import str2bool, shell, read_csv

def main():
    homedir    = shell('echo $HOME')
    featurelst = ' '.join(['rsa', 'thermo', 'onehot', 'pharm', 'hp', 'mass', 'deltar', 'pharm_deltar','hp_deltar', 'msa', 'energy', 'ddg'])
    # ------------------------------------------------------------------------------------------------------------------
    ## parse argument
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_name',       type=str, help='dataset name')
    parser.add_argument('-k', '--k_neighbor', type=int, required=True, nargs='+', help='The feature to append, default=""')
    parser.add_argument('-C', '--center',     type=str, choices=['CA','geometric'], default='CA', help='The MT site center type, default is CA.')
    parser.add_argument('-P', '--PCA',        type=str, default='False', help='if consider PCA, default is False')
    args = parser.parse_args()
    dataset_name  = args.dataset_name
    k_neighborlst = args.k_neighbor
    center        = args.center
    pca           = str2bool(args.PCA)

    QR = QsubRunner(homedir,dataset_name,k_neighborlst,center,pca,featurelst)
    QR.runner()


#-----------------------------------------------------------------------------------------------------------------------
class QsubRunner(object):
    def __init__(self,homedir,dataset_name,k_neighborlst,center,pca,featurelst):
        self.homedir       = homedir
        self.dataset_name  = dataset_name
        self.k_neighborlst = k_neighborlst
        self.center        = center
        self.pca           = pca
        self.feature       = featurelst

        self.app           = '%s/mCNN/src/Spatial/coord.py' % homedir

        self.mt_csv_dir  = '%s/mCNN/dataset/%s/%s.csv' % (homedir, dataset_name, dataset_name)  # after drop duplicates, #PDB,WILD_TYPE,CHAIN,POSITION,MUTANT,PH,TEMPERATURE,DDG
        self.map_csv_dir = '%s/mCNN/dataset/%s/feature/rosetta/global_output/mapping' % (homedir, dataset_name)  # after drop duplicates, #PDB,WILD_TYPE,CHAIN,POSITION,MUTANT,PH,TEMPERATURE,DDG

        self.ref_pdb_dir = '%s/mCNN/dataset/%s/feature/rosetta/ref_output' % (homedir, dataset_name)
        self.mut_pdb_dir = '%s/mCNN/dataset/%s/feature/rosetta/mut_output' % (homedir, dataset_name)

        self.stride_dir  = '%s/mCNN/dataset/%s/feature/stride'%(homedir,dataset_name)
        self.msa_dir     = '%s/mCNN/dataset/%s/feature/msa'%(homedir,dataset_name)

        self.wild_csv_outdir   = '%s/mCNN/dataset/%s/feature/mCNN/wild/csv/' % (homedir, dataset_name)
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

    def runner(self):
        df_mt = read_csv(self.mt_csv_dir)
        df_mt[['POSITION']] = df_mt[['POSITION']].astype(str)#
        for rosetta_mut_tag in os.listdir(self.mut_pdb_dir):
            pdbid, wtaa, chain, pos, _, mtaa = rosetta_mut_tag.split('_')
            mutant_tag = '_'.join([pdbid, wtaa, chain, pos, mtaa])
            try:
                ph, T, ddg = df_mt.loc[(df_mt.PDB==pdbid) & (df_mt.WILD_TYPE==wtaa) & (df_mt.CHAIN==chain) &
                                       (df_mt.POSITION==pos) & (df_mt.MUTANT==mtaa),:].values[0,-3:]
                thermo = str(ph) + ' ' + str(T)
            except:
                print('[ERROR] mutation %s index failed in %s'%(mutant_tag,self.mt_csv_dir))
                sys.exit(1)

            for k_neighbor in self.k_neighborlst:
                self.run_cal_wild(mutant_tag,k_neighbor,thermo,ddg)

                self.run_cal_mutant(rosetta_mut_tag,mutant_tag,k_neighbor,thermo,ddg)

    def run_cal_wild(self,mutant_tag,k_neighbor,thermo,ddg):
        pdbid, wtaa, chain, pos, mtaa = mutant_tag.split('_')
        # -----------------------------parameters for NeighborCalculator-----------------------------
        pdbdir = '%s/%s/%s_ref.pdb' % (self.ref_pdb_dir, pdbid, pdbid)
        # -----------------------------parameters for FeatureGenerator-----------------------------
        wt_blast_dir = '%s/mdl0_wild/%s/%s/msa.cnt_frq.npz'%(self.msa_dir,pdbid,chain)
        mt_blast_dir = '%s/mdl0_mutant/%s/msa.cnt_frq.npz'%(self.msa_dir,mutant_tag)
        energy_dir   = '%s/%s/energy.csv' % (self.ref_pdb_dir, pdbid)
        mapping_dir  = '%s/%s.csv'%(self.map_csv_dir,pdbid)
        sa_dir       = '%s/wild/%s.stride' % (self.stride_dir, pdbid)
        # -----------------------------qsub-----------------------------
        filename = 'center_%s_pca_%s_neighbor_%s'%(self.center,self.pca,k_neighbor)

        qsubid = 'mCNN_wild_%s_%s' % (mutant_tag,filename)
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
        g.writelines('#!/usr/bin/bash\n')
        g.writelines("echo 'user:' `whoami`\necho 'hostname:' `hostname`\necho 'begin at:' `date`\n")
        g.writelines(
            '%s -p %s -tag %s -k %s -c %s -P %s -o %s -n %s -f %s --wtblastdir %s --mtblastdir %s --energydir %s --mappingdir %s -S %s -t %s -d %s\n'
            % (self.app, pdbdir, mutant_tag, k_neighbor, self.center, self.pca, csv_outdir, filename, self.feature, wt_blast_dir,mt_blast_dir,energy_dir, mapping_dir, sa_dir, thermo, ddg))
        g.writelines("echo 'end at:' `date`\n")
        g.close()
        os.system('chmod 755 %s' % run_prog)
        os.system('/public/home/sry/bin/getQ.pl')
        os.system('qsub -e %s -o %s -l %s -N %s %s' % (errfile, outfile, walltime, qsubid, run_prog))
        time.sleep(0.1)

    def run_cal_mutant(self, rosetta_mut_tag,mutant_tag,k_neighbor,thermo,ddg):
        pdbid, wtaa, chain, pos, mtaa = mutant_tag.split('_')
        # -----------------------------parameters for NeighborCalculator-----------------------------
        pdbdir = '%s/%s/%s_mut.pdb' % (self.mut_pdb_dir, rosetta_mut_tag, pdbid)
        # -----------------------------parameters for FeatureGenerator-----------------------------
        wt_blast_dir = '%s/mdl0_wild/%s/%s/msa.cnt_frq.npz' % (self.msa_dir, pdbid, chain)
        mt_blast_dir = '%s/mdl0_mutant/%s/msa.cnt_frq.npz' % (self.msa_dir, mutant_tag)
        energy_dir = '%s/%s/energy.csv' % (self.mut_pdb_dir, rosetta_mut_tag)
        mapping_dir = '%s/%s.csv' % (self.map_csv_dir, pdbid)
        sa_dir = '%s/mutant/%s.stride' % (self.stride_dir, pdbid)
        # -----------------------------qsub-----------------------------
        filename = 'center_%s_pca_%s_neighbor_%s'%(self.center,self.pca,k_neighbor)

        qsubid = 'mCNN_mutant_%s_%s' % (mutant_tag,filename)
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
        g.writelines('#!/usr/bin/bash\n')
        g.writelines("echo 'user:' `whoami`\necho 'hostname:' `hostname`\necho 'begin at:' `date`\n")
        g.writelines(
            '%s -p %s -tag %s -k %s -c %s -P %s -o %s -n %s -f %s --wtblastdir %s --mtblastdir %s --energydir %s --mappingdir %s -S %s -t %s -d %s\n'
            % (self.app, pdbdir, mutant_tag, k_neighbor, self.center, self.pca, csv_outdir, filename, self.feature,
               wt_blast_dir, mt_blast_dir, energy_dir, mapping_dir, sa_dir, thermo, ddg))
        g.writelines("echo 'end at:' `date`\n")
        g.close()
        os.system('chmod 755 %s' % run_prog)
        os.system('/public/home/sry/bin/getQ.pl')
        os.system('qsub -e %s -o %s -l %s -N %s %s' % (errfile, outfile, walltime, qsubid, run_prog))
        time.sleep(0.1)

if __name__ == '__main__':
    main()