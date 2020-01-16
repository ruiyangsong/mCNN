#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, sys, warnings, time, csv
from functools import reduce
import numpy as np
import pandas as pd
from run_rosetta import qsub_ref, qsub_mut
from mCNN.processing import read_csv, PDBparser, aa_123dict, shell, split_tag, log, check_qsub


def main():
    dataset_name, flag = sys.argv[1:]
    HOMEdir = shell('echo $HOME')

    ROS = RosettaEnergy(dataset_name, HOMEdir)
    if flag == 'first':
        ROS.get_mdl0()# mdl0 are needed by others, split two steps for this parallel computing.
    elif flag == 'second':
        ROS.run_ref()
        ROS.mapping_pos()
        ROS.rewrite_mtcsv()
        ROS.run_mut()
        ROS.check_result()
        ROS.get_energy()
        # ROS.test()

class RosettaEnergy(object):
    def __init__(self, dataset_name, homedir):
        self.dataset_name = dataset_name
        self.homedir      = homedir
        self.sleep_time   = 5
        self.check_count  = 0
        self.pdbpath      = '%s/mCNN/dataset/%s/pdb' % (homedir, dataset_name)
        self.refdir       = '%s/mCNN/src/Rosetta/ref.py' % homedir
        self.mutdir       = '%s/mCNN/src/Rosetta/mut.py' % homedir
        self.mt_csv_dir   = '%s/mCNN/dataset/%s/%s.csv' % (homedir, dataset_name,dataset_name)

        self.ref_tagdirlst     = None
        self.mut_tagdirlst     = None
        self.errdir            = '%s/mCNN/dataset/%s/err/rosetta' % (homedir, dataset_name)
        self.outpath_ref       = '%s/mCNN/dataset/%s/feature/rosetta/ref_output' % (homedir, dataset_name)
        self.outpath_mut       = '%s/mCNN/dataset/%s/feature/rosetta/mut_output' % (homedir, dataset_name)
        self.outpath_global    = '%s/mCNN/dataset/%s/feature/rosetta/global_output' % (homedir, dataset_name)
        self.outpath_pdb_mdl0  = '%s/mCNN/dataset/%s/feature/rosetta/global_output/pdb_mdl0' % (homedir, dataset_name)
        self.outpath_mapping   = '%s/mCNN/dataset/%s/feature/rosetta/global_output/mapping' % (homedir, dataset_name)

        if not os.path.exists(self.errdir):
            os.makedirs(self.errdir)
        if not os.path.exists(self.outpath_pdb_mdl0):
            os.makedirs(self.outpath_pdb_mdl0)
        if not os.path.exists(self.outpath_mapping):
            os.makedirs(self.outpath_mapping)
        if not os.path.exists(self.outpath_ref):
            os.makedirs(self.outpath_ref)
        if not os.path.exists(self.outpath_mut):
            os.makedirs(self.outpath_mut)

    @log
    def get_mdl0(self):
        pdbdirlst = [self.pdbpath + '/' + x for x in os.listdir(self.pdbpath)]
        for pdbdir in pdbdirlst:
            PDBparser(pdbdir, MDL=0, write=1, outpath=self.outpath_pdb_mdl0) # only consider standard amino acid residues.
        print('---get mdl0 done!')

    @log
    def run_ref(self):
        '''refine mdl0 file'''
        ref_tag = 'rosetta_ref_%s' % self.dataset_name
        qsub_ref(self.homedir, self.refdir, self.outpath_pdb_mdl0, self.outpath_ref, ref_tag)
        check_qsub(tag = ref_tag, sleep_time = self.sleep_time)
        self.ref_tagdirlst = [self.outpath_ref + '/' + x for x in os.listdir(self.outpath_ref)]

    @log
    def mapping_pos(self):
        '''get the mapping csv for each pdb file, only consider the standard residues'''
        pdbdirlst = [self.outpath_ref + '/' + x +'/'+x+'_ref.pdb' for x in os.listdir(self.outpath_ref)]
        for pdbdir in pdbdirlst:
            chain_id_lst = []
            pos_old_dict = {}
            pos_new_dict = {}
            pdbid = pdbdir.split('/')[-1][0:4]

            model = PDBparser(pdbdir,MDL=0,write=0)

            for chain in model:
                chain_id   = chain.get_id()
                res_id_lst = [res.get_id() for res in chain]
                res_id_lst = list(map(lambda tup: (tup[0] + str(tup[1]) + tup[2]).strip(), res_id_lst))
                if len(res_id_lst) > 0:
                    chain_id_lst.append(chain_id)
                    pos_old_dict[chain_id] = res_id_lst
                else:
                    warnings.warn('WARNING: Chain %s do not contains target atoms.'%chain_id)
            assert len(chain_id_lst) > 0

            chain_len_lst  = [len(pos_old_dict[x]) for x in chain_id_lst]
            df_chain_arr   = []
            df_pos_old_arr = []
            df_pos_new_arr = []
            for i in range(len(chain_id_lst)):
                chain_id = chain_id_lst[i]
                chain_len= chain_len_lst[i]
                if i == 0:
                    pos_new_dict[chain_id] = list(np.arange(1, chain_len + 1))
                if i > 0:
                    cum_sum = reduce(lambda x,y: x+y, chain_len_lst[:i])
                    pos_new_dict[chain_id] = list(np.arange(1, chain_len + 1) + cum_sum)
                df_chain_arr   = df_chain_arr + [chain_id for _ in range(chain_len)]
                df_pos_old_arr = df_pos_old_arr + pos_old_dict[chain_id]
                df_pos_new_arr = df_pos_new_arr + pos_new_dict[chain_id]
            df_mapping = pd.DataFrame({'CHAIN':df_chain_arr, 'POSITION_OLD': df_pos_old_arr, 'POSITION_NEW': df_pos_new_arr})

            df_mapping['CHAIN'] = df_mapping['CHAIN'].astype(str)
            df_mapping['POSITION_OLD'] = df_mapping['POSITION_OLD'].astype(str)
            df_mapping['POSITION_NEW'] = df_mapping['POSITION_NEW'].astype(str)

            df_mapping.to_csv('%s/%s.csv' % (self.outpath_mapping, pdbid), index=False)
        print('---mapping_pos done!')

    @log
    def rewrite_mtcsv(self):
        df = read_csv(self.mt_csv_dir)
        len_df = len(df)
        df.insert(5, 'POSITION_NEW', np.zeros((len_df,1)).astype(object))
        pdbid_last = None
        for i in range(len_df):
            key,pdbid,WILD_TYPE,CHAIN,POSITION,POSITION_NEW,MUTANT = df.iloc[i,:7]
            if pdbid != pdbid_last:
                df_mapping = read_csv('%s/%s.csv' % (self.outpath_mapping, pdbid))
                df_mapping['CHAIN'] = df_mapping['CHAIN'].astype(str)
                df_mapping['POSITION_OLD'] = df_mapping['POSITION_OLD'].astype(str)
                df_mapping['POSITION_NEW'] = df_mapping['POSITION_NEW'].astype(str)
                pdbid_last = pdbid
            pos_new = df_mapping.loc[(df_mapping.CHAIN == CHAIN) & (df_mapping.POSITION_OLD == str(POSITION)), 'POSITION_NEW'].values[0]
            df.iloc[i, 5] = pos_new
        mt_csv_dir_new = '%s/%s.csv' % (self.outpath_global, self.dataset_name)
        df.to_csv(mt_csv_dir_new, index=False)
        print('---rewrite mutant csv file done!')

    @log
    def run_mut(self):
        mt_csv_dir_rewrite = '%s/%s.csv' % (self.outpath_global, self.dataset_name)  ## the rewrite mt_csv
        mut_tag = 'rosetta_mut_%s' % self.dataset_name
        qsub_mut(self.homedir, self.mutdir, self.outpath_ref, self.outpath_mut, mt_csv_dir_rewrite, mut_tag)
        check_qsub(tag=mut_tag, sleep_time=self.sleep_time)
        self.mut_tagdirlst = [self.outpath_mut + '/' + x for x in os.listdir(self.outpath_mut)]

    @log
    def check_result(self):
        ## check failed entries of generate mutant structure
        flag = 0
        dirlst = [self.outpath_mut + '/' + x for x in os.listdir(self.outpath_mut)]
        for dir in dirlst:
            lst = [x for x in os.listdir(dir) if x[-8:] == '_mut.pdb']
            if len(lst) < 1:
                print('ERROR dir of mutant: %s\nmove the directory to %s done!'%(dir,self.errdir))
                os.system('mv %s %s'%(dir,self.errdir))
                self.mut_tagdirlst.remove(dir)
                flag += 1
        if flag > 0:
            print('!!!ERROE of mutant occurs %s time(s)!!!' % flag)

        ## check mapping from pdbid_mut.pdb
        for path in self.mut_tagdirlst:
            tag = split_tag(path)
            pdbid, wtaa, chain, pos_old, pos_new, mtaa = tag.split('_')
            try:
                pos_id = (' ',int(pos_old),' ')
            except:
                # print(pos_old)
                pos_id = (' ',int(pos_old[:-1]), pos_old[-1])
            mt_pdbdir = path+'/%s_mut.pdb'%pdbid
            model = PDBparser(mt_pdbdir,MDL=0,write=0)
            try:
                assert model[chain][pos_id].get_resname() == aa_123dict[mtaa]
            except:
                self.check_count+=1
                print('[ERROR] Check ERROR! Locates at: %s'%path)
        if self.check_count > 0:
            raise RuntimeError('ERROR of mapping occurs %s time(s)!' % self.check_count)
        print('---stay cool with mapping, check rosetta refine and mutant results done!')

    @log
    def get_energy(self):
        for tagdir in self.mut_tagdirlst:
            tag = split_tag(tagdir)
            pdbid = tag.split('_')[0]
            pdbdir = tagdir + '/%s_mut.pdb' % pdbid
            self.write_energy_table(pdbdir, tagdir)

        for tagdir in self.ref_tagdirlst:
            pdbid = tagdir.split('/')[-1]
            pdbdir = tagdir + '/%s_ref.pdb' % pdbid
            self.write_energy_table(pdbdir,tagdir)
        print('---get energy table done!')

    def write_energy_table(self,pdbdir, tagdir):
        f = open(pdbdir, 'r')
        lines = f.readlines()
        f.close()
        f_csv = open('%s/energy.csv' % tagdir, 'w', newline='')
        writer = csv.writer(f_csv, dialect='excel')
        line_index = 0
        label_index = 1000000
        for line in lines:
            line_index += 1
            if line.strip()[:5] == 'label' or line.strip()[:7] == 'weights' or line.strip()[:4] == 'pose':
                label_index = line_index
                writer.writerow(line.split())
            if line.strip()[:3] in list(aa_123dict.values()) and label_index < line_index:
                writer.writerow(line.split())
        f_csv.close()

    def test(self):
        print('function name is:',type(sys._getframe().f_code.co_name))
        print('Test functions')

if __name__ == '__main__':
    main()