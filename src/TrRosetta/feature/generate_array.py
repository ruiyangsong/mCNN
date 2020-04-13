#!/usr/bin/env python
import os, json
import numpy as np
import pandas as pd
from mCNN.processing import aa_123dict
base_TM_dir = '/public/home/sry/mCNN/dataset/TR/feature/TMalign'

def main():
    csv_pth = '/public/home/sry/mCNN/dataset/TR/S2648_TR500.csv'
    df = pd.read_csv(csv_pth)
    for i in range(len(df)):
        key, PDB, WILD_TYPE, CHAIN, POSITION, MUTANT, PH, TEMPERATURE, DDG = df.iloc[i, :].values
        mutant_tag = '%s.%s.%s.%s.%s.%s' % (key, PDB, WILD_TYPE, CHAIN, POSITION, MUTANT)
        # feature_csv_pth = '/public/home/sry/mCNN/dataset/TR/feature/coord/wild_TR/%s_neighbor_20.csv'%mutant_tag
        # TM_json_pth = '/public/home/sry/mCNN/dataset/TR/feature/TMalign/wild_TR_%s/res_dict.json'%mutant_tag
        feature_csv_pth = '/public/home/sry/mCNN/dataset/TR/feature/coord/TR_TR/%s_neighbor_20.csv'%mutant_tag
        TM_json_pth = '/public/home/sry/mCNN/dataset/TR/feature/TMalign/TR_TR_%s/res_dict.json'%mutant_tag
        parse_feature(csvpth=feature_csv_pth,res_json_pth=TM_json_pth)
        break



def parse_feature(csvpth,res_json_pth):
    df = pd.read_csv(csvpth)
    RMSD, TMscore, rotat_mat, trans_vec = parse_TMalign(res_json_pth)

    feature_key = ['dist','x','y','z','s_Helix','s_Strand','s_Coil','sa','rsa','asa','phi','psi',
                   'dist_wild','x_wild','y_wild','z_wild','s_Helix_wild','s_Strand_wild','s_Coil_wild','sa_wild','rsa_wild','asa_wild','phi_wild','psi_wild']
    df_feature = df.loc[:,feature_key]
    wild_coord_arr = df_feature.loc[:, ['x_wild', 'y_wild', 'z_wild']].values
    df_feature.loc[:,['x_wild','y_wild','z_wild']] = np.matmul(wild_coord_arr,rotat_mat)+trans_vec

    # df_feature.loc[:,]
    # mutant_coord_arr = df.loc[:,['wild_x','wild_y','wild_z']]


def parse_TMalign(res_json_pth):
    """
    :param res_json_pth: e.g., ~/mCNN/dataset/TR/feature/TMalign/TR_TR_940.1EY0.S.A.128.A/res_dict.json
                                                                /wild_TR_97.1AMQ.C.A.191.G/res_dict.json
    :return:
    """
    with open(res_json_pth,mode='r') as f:
        res_dict = json.loads(f.readlines()[0])
    RMSD = res_dict['RMSD']
    TMscore = res_dict['Tmscore1']
    mat = np.array(res_dict['mat'])
    rotat_mat = np.transpose(mat[:,1:])
    trans_vec = np.transpose(mat[:,0])
    return RMSD,TMscore,rotat_mat,trans_vec

def subtraction(coord_arr1,coord_arr2,rotate_mat,trans_vec):
    coord_arr1 = np.matmul(coord_arr1,rotate_mat)+trans_vec
    return coord_arr1 - coord_arr2

if __name__ == '__main__':
    main()