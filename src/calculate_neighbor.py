#!～/anaconda3/envs/bioinfo/bin/python
# -*- coding: utf-8 -*-

# file_name : calculate_neighbor.py
# time      : 3/06/2019 16:12
# author    : ruiyang
# email     : ww_sry@163.com

import sys
import numpy as np
import pandas as pd
import warnings
from Bio import BiopythonWarning
from Bio.PDB.PDBParser import PDBParser
from sklearn.decomposition import PCA
from generate_dataset import get_df_dataset, generate_dataset, save_data_array


def transform(df_neighbor_mutation, center_coord, model=0, center_CA=True):
    '''
    :param df_neighbor_mutation: pandas DataFrame which stored neighbor information of this mutation.
    :param center_coord: center coords of this neighborhood.
    :param model: different model to transform neighboring atom coords.
     [0,1,2] = [NormalPCA, TensorPCA, First two lying on Peptide plane], model 0 is the default.
    :param center_CA: bool, if the center is Carbon_alpha.
    :return: pandas DataFrame which stored neighbor information (after transform of coords) of this mutation.
    '''
    coord_array_before = df_neighbor_mutation.loc[:, ['x', 'y', 'z']].values  # numpy array.
    #print('coords format is：\n',coord_array_before.dtype) #float
    assert len(coord_array_before) >= 3 #row number.
    if model == 0:
        pca = PCA(n_components = 3)
        pca.fit(coord_array_before)
        ## full PCA
        coord_array_after = pca.transform(coord_array_before)
        center_coord_after = pca.transform(center_coord.reshape(-1,3))
        ## half PCA
        # base_vec = pca.components_
        # coord_array_after = np.dot(coord_array_before,np.transpose(base_vec))#旋转之后的坐标数组
        # center_coord_after = np.dot(center_coord,np.transpose(base_vec))#旋转之后的中心坐标数组
    elif model == 1:
        pass
    elif model == 2:
        pass

    if center_CA:
        # mv center_CA to (0,0,0).
        coord_array_after = coord_array_after - center_coord_after

    df_neighbor_mutation.loc[:, ['x', 'y', 'z']] = coord_array_after
    return df_neighbor_mutation


def get_neighbor(df_mutation_protein, radius, path_pdbfile_base, k_neighbor=0, only_atom=True, centerCA=True, pca_model=0):
    '''
    :param df_mutation_protein: The pandas DataFrame stored mutation information of this protein (pdbid).
    :param radius: float, neighboring radius.
    When k_neighbor != 0, radius should set big enough to confirm the environment contains at least k atoms.
    :param path_pdbfile_base: str, base path of the pdb file.
    :param k_neighbor: int, k neighboring atoms when environment denoted by k_neighbor.
    :param only_atom: bool, whether consider the protein atoms only. Default is True.
    :param centerCA: bool, whether the center is carbon_alpha. Default is True.
    :param pca_model: int [0,1,2], parameter send to function transform. refers to model parameter in function transform.
    :return: pandas DataFrame, stored the neighbor environment of this protein, which may have more than one mutation,
     their coords have been transformed.
    '''
    df_neighbor_protein = pd.DataFrame(
        {'key':[],'pdb': [], 'wild_type': [], 'chain': [], 'position': [], 'inode': [],'rsa':[], 'mutant': [], 'ph': [],
         'temperature': [], 'ddg': [], 'full_name': [], 'name': [], 'dist': [], 'x': [], 'y': [], 'z': []}) # 包含此pdb结构所有邻域信息的DataFrame.

    num_mutation_protein = len(df_mutation_protein) # number of mutations in this protein.
    pdbid = df_mutation_protein.iloc[0, 1][0:4]
    #print('The pdbid is: %s, contains %d mutations.' % (pdbid, num_mutation_protein))

    ## ==================== parser this protein structure. ====================
    warnings.simplefilter('ignore', BiopythonWarning)
    parser = PDBParser(PERMISSIVE=1)
    structure = parser.get_structure(pdbid, path_pdbfile_base + '/' + pdbid + '.pdb')
    for i in range(num_mutation_protein):
        df_neighbor_mutation = pd.DataFrame(
            {'key':[], 'pdb': [], 'wild_type': [], 'chain': [], 'position': [], 'inode': [],'rsa':[], 'mutant': [], 'ph': [],
             'temperature': [], 'ddg': [], 'full_name': [], 'name': [], 'dist': [], 'x': [], 'y': [], 'z': []})

        row = df_mutation_protein.iloc[i, :]
        inode = ' '
        try:
            int(row.POSITION)
        except:
            # print(type(row.POSITION))# str
            inode = row.POSITION[-1:]
            print('pdbid: %s, inode: %s' % (row.PDB[0:4], row.POSITION[-1:]))
            row.POSITION = row.POSITION[0:-1]
        try:
            rsa = row.RSA
        except:
            rsa = 0
        [key, pdb, wild_type, chain, position, inode, mutant, ph, temperature, ddg] = \
            [row.key, row.PDB[0:4], row.WILD_TYPE, row.CHAIN, int(row.POSITION), inode, row.MUTANT, float(row.PH),
             float(row.TEMPERATURE), float(row.DDG)]
        # print([pdb,wild_type,chain,position,mutant,ph,temperature,ddg])

        ## ==================== calculate the center coords of neighbors (considering inode!!!) ====================
        if centerCA:
            #print('The center is: alpha_C.')
            if inode == ' ':
                center_coord = structure[0][chain][position]['CA'].get_coord()
            else:
                center_coord = structure[0][chain][(' ',position,inode)]['CA'].get_coord()
                #print('inode is:',inode)
            # print(type(center_coord))
        else:
            #print('The center is: geometric chenter.)
            center_coord = np.array([0, 0, 0])
            if inode == ' ':
                residue = structure[0][chain][position]
            else:
                residue = structure[0][chain][(' ',position,inode)]
            for atom in residue:
                center_coord = center_coord + atom.get_coord()
            center_coord = center_coord / len(residue)
        #print('The center coords are：%r' % center_coord)

        ## ==================== calculate the neighboring environment. ====================
        if only_atom:
            ## only consider standard atoms in protein.
            residues = structure[0].get_residues()
            for residue in residues:
                hetfield = residue.get_id()[0]
                if hetfield == ' ':
                    for atom in residue:
                        atom_coord = atom.get_coord()
                        full_name = atom.get_name()
                        name = full_name[0:1]
                        dist = np.linalg.norm(center_coord - atom_coord)

                        if dist <= radius:
                            temp_array = np.array(
                                [key, pdb, wild_type, chain, position, inode, rsa, mutant, ph, temperature, ddg,
                                 full_name, name, dist, atom_coord[0], atom_coord[1], atom_coord[2]])
                            temp_df = pd.DataFrame(temp_array.reshape(1, len(temp_array)))
                            temp_df.columns = df_neighbor_mutation.columns
                            df_neighbor_mutation = pd.concat([df_neighbor_mutation, temp_df], axis=0, ignore_index=True)
        else:
            # consider all atoms.
            atoms = structure.get_atoms()
            for atom in atoms:
                atom_coord = atom.get_coord()
                full_name = atom.get_name()
                name = full_name[0:1]
                dist = np.linalg.norm(center_coord - atom.get_coord())

                if dist <= radius:
                    temp_array = np.array(
                        [key, pdb, wild_type, chain, position, inode, rsa, mutant, ph, temperature, ddg, full_name, name,
                         dist, atom_coord[0], atom_coord[1], atom_coord[2]])
                    temp_array = temp_array.reshape(1, len(temp_array))
                    temp_df = pd.DataFrame(temp_array)
                    temp_df.columns = df_neighbor_mutation.columns
                    df_neighbor_mutation = pd.concat([df_neighbor_mutation, temp_df], axis=0, ignore_index=True)

        # ==================== set data format of each column ====================
        #print('These foramts before setting are:\n', df_neighbor_mutation.dtypes)
        df_neighbor_mutation[['key']] = df_neighbor_mutation[['key']].astype(int)
        df_neighbor_mutation[['position']] = df_neighbor_mutation[['position']].astype(int)
        df_neighbor_mutation[['ph']] = df_neighbor_mutation[['ph']].astype(float)
        df_neighbor_mutation[['temperature']] = df_neighbor_mutation[['temperature']].astype(float)
        df_neighbor_mutation[['ddg']] = df_neighbor_mutation[['ddg']].astype(float)
        df_neighbor_mutation[['dist']] = df_neighbor_mutation[['dist']].astype(float)
        df_neighbor_mutation[['x']] = df_neighbor_mutation[['x']].astype(float)
        df_neighbor_mutation[['y']] = df_neighbor_mutation[['y']].astype(float)
        df_neighbor_mutation[['z']] = df_neighbor_mutation[['z']].astype(float)
        df_neighbor_mutation[['rsa']] = df_neighbor_mutation[['rsa']].astype(float)
        #print('These foramts after setting are:\n', df_neighbor_mutation.dtypes)
        #print('before sort',df_neighbor_mutation)
        df_neighbor_mutation = df_neighbor_mutation.sort_values(by='dist', ascending = True)
        #print('after sort',df_neighbor_mutation)

        ## when environment are chosen as k_neighbir.
        if k_neighbor != 0:
            assert len(df_neighbor_mutation) >= k_neighbor
            df_neighbor_mutation = df_neighbor_mutation.iloc[0:k_neighbor, :]

        df_neighbor_mutation_transform = transform(df_neighbor_mutation, center_coord, model = pca_model, center_CA = centerCA)
        df_neighbor_protein = pd.concat([df_neighbor_protein, df_neighbor_mutation_transform])
    return df_neighbor_protein


if __name__ == '__main__':
    ## input parameters in shell.
    name_dataset, radius, k_neighbor, num_class = sys.argv[1:]
    radius = float(radius)
    k_neighbor = int(k_neighbor)
    num_class = int(num_class)

    path_dataset = '../datasets/%s/' % name_dataset
    path_csv_mutation = path_dataset + '%s_new.csv' % name_dataset # path of the .csv file which describes the mutant\
    # information of this dataset.

    ## input parameters of function get_neighbor.
    path_pdbfile_base = path_dataset + 'pdb' + name_dataset
    only_atom = True
    centerCA = True
    pca_model = 0

    f = open(path_csv_mutation, 'r'); df_mutation = pd.read_csv(f); f.close()
    list_name_protein = list(df_mutation.drop_duplicates('PDB', 'first', inplace=False).PDB)
    #print(list_name_protein)
    num_mutation, num_protein = len(df_mutation), len(list_name_protein)
    print('This dataset is：%s, Contains %d mutations, In %d proteins.'
          % (name_dataset, num_mutation, num_protein))

    ## Initialize the columns of .csv file which deposits the neighbor atom coordinates. 2D array.
    df_coord_all = pd.DataFrame(
        {'key':[], 'pdb': [], 'wild_type': [], 'chain': [], 'position': [], 'inode':[], 'rsa':[], 'mutant': [], 'ph': [],
         'temperature': [], 'ddg': [], 'full_name': [], 'name': [], 'dist': [], 'x': [], 'y': [], 'z': []})

    ## loop for storing every pdbid (mutations in this protein) into the df_coord_all.
    for name_protein in list_name_protein:
        df_mutation_protein = df_mutation[df_mutation.PDB == name_protein] # example of name_protein: xxxx.pdb.
        df_coord_protein = get_neighbor(df_mutation_protein, radius, path_pdbfile_base, k_neighbor=k_neighbor,
                                        only_atom = only_atom, centerCA=centerCA, pca_model=pca_model)
        df_coord_all = pd.concat([df_coord_all, df_coord_protein])

    path_csv_coord = path_dataset + name_dataset + '_r_%.2f_neighbor_%d_onlyAtom_%s_centerCA_%s.csv'\
                     % (radius, k_neighbor, only_atom, centerCA)
    df_coord_all.to_csv(path_csv_coord, index=False)
    print('store df_coord_all done.', 'Shape of df_coord_all:', df_coord_all.shape, '\n', 'begin generate datasets_array...')

    ## ==================== Call generate_dataset.py ====================
    df_delta_r = get_df_dataset(df_coord_all, atom_class=num_class)
    ##计算并保存x,y
    x, y, ddg_value = generate_dataset(num_mutation, df_delta_r, cutoff_step=0, k_neighbor=k_neighbor)
    save_data_array(x, y, ddg_value, name_dataset, radius, k_neighbor, num_class)