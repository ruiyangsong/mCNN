#!/usr/bin/env python
import os, json
import numpy as np
import pandas as pd
base_TM_dir = '/public/home/sry/mCNN/dataset/TR/feature/TMalign'

def main():
    csv_pth = '/public/home/sry/mCNN/dataset/TR/S2648_TR500_with_PROB.csv'
    df = pd.read_csv(csv_pth)
    outdir = '/public/home/sry/mCNN/dataset/TR/npz'
    filename = 'neighbor_20'

    x_lst = []
    ddg_lst = []

    for i in range(len(df)):
        key, PDB, WILD_TYPE, CHAIN, POSITION, MUTANT, PH, TEMPERATURE, DDG, WILD_PROB, MUTANT_PROB = df.iloc[i, :].values
        mutant_tag = '%s.%s.%s.%s.%s.%s' % (key, PDB, WILD_TYPE, CHAIN, POSITION, MUTANT)
        # feature_csv_pth = '/public/home/sry/mCNN/dataset/TR/feature/coord/wild_TR/%s_neighbor_20.csv'%mutant_tag
        # TM_json_pth = '/public/home/sry/mCNN/dataset/TR/feature/TMalign/wild_TR_%s/res_dict.json'%mutant_tag
        feature_csv_pth = '/public/home/sry/mCNN/dataset/TR/feature/coord/TR_TR/%s_neighbor_20.csv'%mutant_tag
        TM_json_pth = '/public/home/sry/mCNN/dataset/TR/feature/TMalign/TR_TR_%s/res_dict.json'%mutant_tag
        feature_arr = parse_feature(csvpth=feature_csv_pth,res_json_pth=TM_json_pth, subtract_prob=WILD_PROB-MUTANT_PROB)
        x_lst.append(feature_arr)
        ddg_lst.append(DDG)
    os.makedirs(outdir,exist_ok=True)
    np.savez('%s/%s.npz'%(outdir,filename),x=np.array(x_lst),ddg=np.array(ddg_lst))

def parse_feature(csvpth,res_json_pth,subtract_prob):
    df = pd.read_csv(csvpth)
    len_df = len(df)
    with open(res_json_pth,mode='r') as f:
        res_dict = json.loads(f.readlines()[0])
    RMSD = res_dict['RMSD']
    TMscore = res_dict['Tmscore1']
    mat = np.array(res_dict['mat'])
    rotat_mat = np.transpose(mat[:,1:])
    trans_vec = np.transpose(mat[:,0])

    feature_key = ['dist', 'x', 'y', 'z', 's_Helix', 's_Strand', 's_Coil', 'sa', 'rsa', 'asa', 'phi', 'psi',
                   'dist_wild', 'x_wild', 'y_wild', 'z_wild', 's_Helix_wild', 's_Strand_wild', 's_Coil_wild',
                   'sa_wild', 'rsa_wild', 'asa_wild',
                   'phi_wild', 'psi_wild']
    df_feature = df.loc[:, feature_key]
    wild_coord_arr = df_feature.loc[:, ['x_wild', 'y_wild', 'z_wild']].values
    df_feature.loc[:, ['x_wild', 'y_wild', 'z_wild']] = np.matmul(wild_coord_arr, rotat_mat) + trans_vec

    wild_dist_arr = df_feature.loc[:, 'dist_wild'].values
    wild_coord_arr = df_feature.loc[:, ['x_wild', 'y_wild', 'z_wild']].values
    wild_secondary_arr = df_feature.loc[:, ['s_Helix_wild', 's_Strand_wild', 's_Coil_wild']].values
    wild_rsa_arr = df_feature.loc[:, 'rsa_wild'].values
    wild_phi_arr = df_feature.loc[:, 'phi_wild'].values
    wild_psi_arr = df_feature.loc[:, 'psi_wild'].values

    mutant_dist_arr = df_feature.loc[:, 'dist'].values
    mutant_coord_arr = df_feature.loc[:, ['x', 'y', 'z']].values
    mutant_secondary_arr = df_feature.loc[:, ['s_Helix', 's_Strand', 's_Coil']].values
    mutant_rsa_arr = df_feature.loc[:, 'rsa'].values
    mutant_phi_arr = df_feature.loc[:, 'phi'].values
    mutant_psi_arr = df_feature.loc[:, 'psi'].values

    subtract_dist_arr = np.array([wild_dist_arr[i]-mutant_dist_arr[i] for i in range(len_df)]).reshape(len_df,-1)
    coord_dist_arr = np.array([[np.linalg.norm(wild_coord_arr[i]-mutant_coord_arr[i]) for i in range(len_df)]]).reshape(len_df,-1)
    subtract_secondary_arr = np.array([wild_secondary_arr[i]-mutant_secondary_arr[i] for i in range(len_df)]).reshape(len_df,-1)
    subtract_rsa_arr = np.array([wild_rsa_arr[i]-mutant_rsa_arr[i] for i in range(len_df)]).reshape(len_df,-1)
    subtract_sin_phi_arr = np.array([np.sin(wild_phi_arr[i])-np.sin(mutant_phi_arr[i]) for i in range(len_df)]).reshape(len_df,-1)
    subtract_cos_phi_arr = np.array([np.cos(wild_phi_arr[i])-np.cos(mutant_phi_arr[i]) for i in range(len_df)]).reshape(len_df,-1)
    subtract_sin_psi_arr = np.array([np.sin(wild_psi_arr[i]) - np.sin(mutant_psi_arr[i]) for i in range(len_df)]).reshape(len_df,-1)
    subtract_cos_psi_arr = np.array([np.cos(wild_psi_arr[i]) - np.cos(mutant_psi_arr[i]) for i in range(len_df)]).reshape(len_df,-1)


    RMSD_arr = np.array((RMSD,)*len_df).reshape(len_df,-1)
    TMscore_arr = np.array((TMscore,)*len_df).reshape(len_df,-1)
    subtract_prob_arr = np.array((subtract_prob,)*len_df).reshape(len_df,-1)

    feature_arr = np.hstack((subtract_dist_arr,coord_dist_arr,subtract_secondary_arr,subtract_rsa_arr,subtract_sin_phi_arr,
                             subtract_cos_phi_arr,subtract_sin_psi_arr,subtract_cos_psi_arr,RMSD_arr,TMscore_arr,subtract_prob_arr))
    # print(feature_arr.shape)

    return feature_arr

if __name__ == '__main__':
    main()