#!/usr/local/bin/python
# -*- coding: utf-8 -*-
import csv
import os, sys, warnings
from functools import reduce
import numpy as np
import pandas as pd
from processing import read_csv, df2csv, PDBparser, aa_123dict, shell, split_tag

def main():
    dataset_name = sys.argv[1]
    HOMEdir = shell('echo $HOME')
    pdbpath = '%s/mCNN/datasets/%s/pdb%s' % (HOMEdir,dataset_name,dataset_name)
    outpath = '%s/mCNN/datasets/%s/ref%s' % (HOMEdir,dataset_name, dataset_name)
    pdbdirlst = [pdbpath + '/' + x for x in os.listdir(pdbpath) if os.path.isfile(pdbpath + '/'+x)]

    ROS = RosettaEnergy(dataset_name,HOMEdir)
    ROS.mapping_pos(*pdbdirlst)
    ROS.rewrite_mtcsv()
    # ROS.get_ref(app, pdbpath, outpath)

class RosettaEnergy(object):
    def __init__(self, dataset_name, homedir):
        self.dataset_name = dataset_name
        self.homedir      = homedir
        self.appdir       = '%s/mCNN/TrRosetta/ref.py' % homedir
        self.mt_csv_path  = '%s/mCNN/datasets/%s' % (homedir, dataset_name)
        self.mapping_path = '%s/mCNN/datasets/%s/pos_mapping' % (homedir, dataset_name)
        if not os.path.exists(self.mapping_path):
            os.system('mkdir -p %s'%self.mapping_path)

    def mapping_pos(self, *pdbdirtup):
        '''
        * rename the residue serial number of original pdb files. *
        :param pdbdir: list of pdb file directory
        :return:       saved mapping csvs of pdb file
        '''
        for pdbdir in pdbdirtup:
            chain_id_lst = []
            pos_old_dict = {}
            pos_new_dict = {}
            pdbid = pdbdir[-8:-4]
            model = PDBparser(pdbdir, 0,write=1)
            for chain in model:
                chain_id   = chain.get_id()
                res_id_lst = [res.get_id() for res in chain]
                res_id_lst = list(filter(lambda tup: tup[0] != 'W', res_id_lst)) # PyRosetta consider the NUCLEIC res but not HETATM_W res(HOH)
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
            df_mapping['POSITION_NEW'] = df_mapping['POSITION_NEW'].astype(str)
            df2csv(df=df_mapping, csvdir = '%s/%s.csv' % (self.mapping_path, pdbid))

    def rewrite_mtcsv(self):
        mtcsv_old_dir = self.mt_csv_path + '/' + self.dataset_name + '_new.csv'
        mtcsv_new_dir = self.mt_csv_path + '/' + self.dataset_name + '_new_map.csv'
        df = read_csv(mtcsv_old_dir)
        len_df = len(df)
        df.insert(5, 'POSITION_NEW', np.zeros((len_df,1)).astype(object))
        pdbid_last = 'tmp'
        df_mapping = 'tmpdf'
        for i in range(len_df):
            key,PDB,WILD_TYPE,CHAIN,POSITION,POSITION_NEW,MUTANT = df.iloc[i,:7]
            if self.dataset_name == 'S1925':
                pdbid = PDB
            elif self.dataset_name == 'S2648':
                pdbid = PDB[:-5]
            if pdbid != pdbid_last:
                df_mapping = read_csv('%s/%s.csv' % (self.mapping_path, pdbid))
                df_mapping['CHAIN'] = df_mapping['CHAIN'].astype(str)
                df_mapping['POSITION_OLD'] = df_mapping['POSITION_OLD'].astype(str)
                df_mapping['POSITION_NEW'] = df_mapping['POSITION_NEW'].astype(str)
                pdbid_last = pdbid
            try:
                pos_new = df_mapping.loc[(df_mapping.CHAIN == CHAIN) & (df_mapping.POSITION_OLD == str(POSITION)), 'POSITION_NEW'].values[0]
            except:
                print('PDB:%s,WILD_TYPE:%s,CHAIN:%s,POSITION:%s'%(PDB,WILD_TYPE,CHAIN,POSITION))
            df.iloc[i, 5] = pos_new
        df2csv(df=df,csvdir=mtcsv_new_dir)


    def get_ref(self, pdbpath, outpath):
        mtcsv_new_dir = self.mt_csv_path + '/' + self.dataset_name + '_new_map.csv'
        df = read_csv(mtcsv_new_dir)
        for i in range(len(df)):
            key,PDB,WILD_TYPE,CHAIN,POSITION,POSITION_NEW,MUTANT = df.iloc[i,:7]
            pdbid = PDB[:-5]
            mt_aa = aa_123dict[MUTANT]
            pdbdir = '%s/%s.pdb' % (pdbpath, pdbid)
            tag = '%s_%s_%s_%s_%s_%s' % (pdbid, WILD_TYPE, CHAIN, POSITION, POSITION_NEW, MUTANT)
            subpath = '%s/%s' % (outpath, tag)
            if not os.path.exists(subpath):
                os.system('mkdir -p %s' % subpath)
            os.system('cp %s %s' % (pdbdir, subpath))
            os.chdir(subpath)
            os.system('%s %s %s %s\n' % (self.appdir, pdbid, POSITION_NEW, mt_aa))

    def check_result(self, *tagpath):
        count = 0
        for path in tagpath:
            tag = split_tag(path)
            pdbid, wtaa, chain, pos_old, pos_new, mtaa = tag.split('_')
            try:
                pos_id = (' ',int(pos_old),' ')
            except:
                print(pos_old)
                pos_id = (' ',int(pos_old[:-1]),pos_old[-1])
            mt_pdbdir = path+'/%s_mut.pdb'%pdbid
            model = PDBparser(mt_pdbdir, 0)
            try:
                assert model[chain][pos_id].get_resname() == aa_123dict[mtaa]
            except:
                count+=1
                print('[ERROR] Check ERROR! Locates at: %s'%path)
        print('ERROR occurs %s time(s)!' % count)

    def get_energy(self, *tagpath):
        for path in tagpath:
            tag = split_tag(path)
            pdbid = tag.split('_')[0]
            for flag in ['ref', 'mut']:
                pdbdir = path + '/%s_%s.pdb' % (pdbid,flag)
                f = open(pdbdir, 'r')
                lines = f.readlines()
                f.close()
                f_csv = open('%s/energy_%s.csv'%(path,flag), 'w', newline='')
                writer = csv.writer(f_csv, dialect='excel')
                for line in lines:
                    if line.strip()[:3] in list(aa_123dict.values()):
                        writer.writerow(line.split())
                f_csv.close()

    def test(self):
        print('function name is:',type(sys._getframe().f_code.co_name))
        print('Test functions')

if __name__ == '__main__':
    main()
