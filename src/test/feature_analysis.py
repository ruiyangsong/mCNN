#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, sys
from mCNN.processing import read_csv, calc_coor_pValue

dataset_name = sys.argv[1]

def ddg_energy_coor():
    '''计算数据集中ddg 和 Rosetta突变前后 total 能量变化 之间的相关系数'''
    mut_csv_dir    = '/public/home/sry/mCNN/dataset/%s/%s.csv'%(dataset_name,dataset_name)
    mut_output_dir = '/public/home/sry/mCNN/dataset/%s/feature/rosetta/mut_output'%dataset_name
    ref_output_dir = '/public/home/sry/mCNN/dataset/%s/feature/rosetta/ref_output'%dataset_name
    mut_tag_dir    = [mut_output_dir+'/'+x for x in os.listdir(mut_output_dir)]

    mut_energy_total = [] # mut energy by ref.py and mut.py
    ref_energy_total = [] # ref energy by ref.py
    ddg_total        = [] # ddg values from data sets
    mt_df = read_csv(mut_csv_dir)
    mt_df[['POSITION']] = mt_df[['POSITION']].astype(str)
    for tagdir in mut_tag_dir:
        pdbid,wtaa,chain,pos_old,pos_new,mtaa = tagdir.split('/')[-1].split('_')
        with open('%s/%s_mut.sc'%(tagdir,pdbid), 'r') as f:
            mut_energy_total.append(float(f.read().split()[-1]))
        with open('%s/%s/%s_ref.sc'%(ref_output_dir,pdbid,pdbid), 'r') as f:
            ref_energy_total.append(float(f.read().split()[-1]))
        try:
            ddg = mt_df.loc[(mt_df.PDB ==pdbid) & (mt_df.WILD_TYPE == wtaa) & (mt_df.CHAIN == chain)
                            & (mt_df.POSITION == pos_old) & (mt_df.MUTANT == mtaa), 'DDG'].values[0]
            ddg_total.append(ddg)
        except:
            print('[ERROR] ddg retrieve failed, tag dir is: %s'%tagdir)
            sys.exit(1)

    ddg_energy_total = [i-j for i,j in zip(mut_energy_total,ref_energy_total)] #rosetta total energy change

    pearson_coeff, p_value = calc_coor_pValue(ddg_total,ddg_energy_total)

    return pearson_coeff, p_value

if __name__ == '__main__':
    pearson_coeff, p_value = ddg_energy_coor()
    print('相关系数：%s\np值：%s'%(pearson_coeff, p_value))