#!～/anaconda3/env/bioinfo/bin/python
# -*- coding: utf-8 -*-

# file_name : calculate_neighbor.py
# time      : 3/06/2019 16:12
# author    : ruiyang
# email     : ww_sry@163.com

import sys
import numpy as np
import pandas as pd
from Bio.PDB.PDBParser import PDBParser
from sklearn.decomposition import PCA


def transform(df_init, center_coord, model=0):
    """
    参数:
            df_init     : 存储“单点突变信息”的DataFrame，为了确定肽平面使用，模型3被选择时，此参数必须传入.
            center_coord: 中心坐标
            model       : pca选择的模型：0 - 使用普通PCA(一个主方向是一个向量方向)将原始坐标数据旋转.
                                          1 - 使用张量PCA(一个主方向是一个张量)将原始坐标数据旋转.
                                          3 - 确定肽平面，然后数据点向肽平面投影，然后做PCA得到三个方向，进而将原始坐标数据旋转.
    函数功能：
            将输入的相对坐标数据在确定的新坐标系下表示（由PCA确定坐标系，并将中心平移至原点）.
    调用：
            df_init_after = get_neighbor(df_init, center_coord, model=0)
    """
    coord_array_before = df_init.loc[:, ['x', 'y', 'z']].values  # 存储单点突变邻域原子坐标的numpy数组
    #print('坐标格式为：\n',coord_array_before.dtype) #float
    assert len(coord_array_before) >= 3
    if model == 0:
        pca = PCA(n_components = 3)
        pca.fit(coord_array_before)
        ## full PCA
        # coord_array_after_r = pca.transform(coord_array_before)
        # center_coord_after_r = pca.transform(center_coord.reshape(-1,3))
        ## half PCA
        base_vec = pca.components_
        coord_array_after_r = np.dot(coord_array_before,np.transpose(base_vec))#旋转之后的坐标数组
        center_coord_after_r = np.dot(center_coord,np.transpose(base_vec))#旋转之后的中心坐标数组
    elif model == 1:
        pass
    elif model == 2:
        pass
    # 将坐标平移
    transform_vec = np.array([0,0,0]) - center_coord_after_r   # 平移的向量
    coord_array_after = coord_array_after_r + transform_vec  # 平移之后的坐标数组
    df_init.loc[:, ['x', 'y', 'z']] = coord_array_after

    return df_init


def get_neighbor(mutations_df_pdbid, radius, pdbfile_base_path, k_neighbor=0, only_atom=True, centerCA=True, pca_model=0):
    '''
    参数：     mutations_df_pdbid: “一个pdb结构”的所有突变信息，其数据结构为一个DataFrame，每一行表示一个突变信息.
              radius            ：邻域半径.
              pdbfile_base_path : 存储pdb文档的地址.
              k_neighbor        : 最近的 k 个邻居,默认值为0（氨基酸最多包含原子个数为14）.
              only_atom         : 是否只考虑标准残基中的原子，默认为True.
              centerCA          : 以alpha C 为残基中心，默认为True.

    函数功能：将一个结构包含的所有突变信息的（由相对坐标描述的）邻域环境写入一个大的df返回.

    流程描述：根据传入的信息（DataFrame）读取pdb文件。针对此结构上的“第k个突变信息（k遍历所有突变）”计算“邻域环境”（调用了Bio.PDB）,
              将其写入一个dataframe子集.最后将这些自己合并为一个dataframe，df前加入的是标识突变的主键，最终的结构为:
              -----------
              df.columns = ['key','pdb','wild_type','chain','position','inode','mutant','ph','temperature','ddg',
              '原子全名(full_name)','原子名称(name)','各原子距离中心的距离(dist)',x','y','z'].
              -----------
              后期会基于返回的df进行旋转和坐标平移：将突变位点的中心(alpha C 或 突变残基的几何中心)平移到原点.

    调用：    df_neighbor = get_neighbor(mutations_df_pdbid,radius,pdbfile_base_path,only_atom=True,centerCA=True)

    '''
    df_neighbor = pd.DataFrame(
        {'key':[],'pdb': [], 'wild_type': [], 'chain': [], 'position': [], 'inode': [],'rsa':[], 'mutant': [], 'ph': [],
         'temperature': [], 'ddg': [], 'full_name': [], 'name': [], 'dist': [], 'x': [], 'y': [], 'z': []}) # 包含此pdb结构所有邻域信息的DataFrame.

    num_of_mutations = len(mutations_df_pdbid)
    pdbid = mutations_df_pdbid.iloc[0, 1][0:4]
    print('此结构的id为：%s，包含%d个突变' % (pdbid, num_of_mutations))
    # ==================== 解析此结构 ====================
    parser = PDBParser(PERMISSIVE=1)
    structure = parser.get_structure(pdbid, pdbfile_base_path + '/' + pdbid + '.pdb')
    for i in range(num_of_mutations):
        # ==================== 对结构中的每一个突变进行以下操作 ====================
        df_init = pd.DataFrame(
            {'key':[], 'pdb': [], 'wild_type': [], 'chain': [], 'position': [], 'inode': [],'rsa':[], 'mutant': [], 'ph': [],
             'temperature': [], 'ddg': [], 'full_name': [], 'name': [], 'dist': [], 'x': [], 'y': [], 'z': []})  # 存储一个突变的df，最终将其汇总在df_neighbor

        row = mutations_df_pdbid.iloc[i, :]

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
        [key, pdb, wild_type, chain, position, inode, mutant, ph, temperature, ddg] = [row.key, row.PDB[0:4], row.WILD_TYPE,
                                                                                       row.CHAIN, int(row.POSITION), inode,
                                                                                       row.MUTANT, float(row.PH),
                                                                                       float(row.TEMPERATURE), float(row.DDG)]
        # print([pdb,wild_type,chain,position,mutant,ph,temperature,ddg])

        # ==================== 计算突变位点的几何中心(注：考虑inode!!!) ====================
        if centerCA:
            #print('中心选取的是：alpha_C.')
            if inode == ' ':
                center_coord = structure[0][chain][position]['CA'].get_coord()
            else:
                center_coord = structure[0][chain][(' ',position,inode)]['CA'].get_coord()
                #print('inode is:',inode)
            # print(type(center_coord))
        else:
            #print('中心选取的是：残基的几何中心.')
            center_coord = np.array([0, 0, 0])
            if inode == ' ':
                residue = structure[0][chain][position]
            else:
                residue = structure[0][chain][(' ',position,inode)]
            for atom in residue:
                center_coord = center_coord + atom.get_coord()
            center_coord = center_coord / len(residue)

        #print('此突变位点中心坐标为：%r' % center_coord)

        # ==================== 计算中心的邻域环境 ====================
        if only_atom:
            # 邻域只考虑标准残基原子
            residues = structure[0].get_residues()
            for residue in residues:
                hetfield = residue.get_id()[0]
                if hetfield == ' ':
                    for atom in residue:
                        atom_coord = atom.get_coord()
                        full_name = atom.get_name()
                        name = full_name[0:1]

                        dist = np.linalg.norm(center_coord - atom_coord)
                        # # 邻域选为K近邻
                        # if radius == 0:
                        #     temp_array = np.array(
                        #         [pdb, wild_type, chain, position, inode, mutant, ph, temperature, ddg, full_name, name,
                        #          dist, atom_coord[0], atom_coord[1], atom_coord[2]])
                        #     temp_array = temp_array.reshape(1, len(temp_array))
                        #     temp_df = pd.DataFrame(temp_array)
                        #     temp_df.columns = df_init.columns
                        #     df_init = pd.concat([df_init, temp_df], axis=0, ignore_index=True)
                        # 邻域根据半径计算
                        if dist <= radius:
                            temp_array = np.array(
                                [key, pdb, wild_type, chain, position, inode, rsa, mutant, ph, temperature, ddg, full_name, name,
                                 dist, atom_coord[0], atom_coord[1], atom_coord[2]])
                            temp_array = temp_array.reshape(1, len(temp_array))
                            temp_df = pd.DataFrame(temp_array)
                            temp_df.columns = df_init.columns
                            df_init = pd.concat([df_init, temp_df], axis=0, ignore_index=True)
        else:
            # 邻域考虑所有原子
            atoms = structure.get_atoms()
            for atom in atoms:
                atom_coord = atom.get_coord()
                full_name = atom.get_name()
                name = full_name[0:1]
                dist = np.linalg.norm(center_coord - atom.get_coord())
                # if radius == 0:
                #     temp_array = np.array(
                #         [pdb, wild_type, chain, position, inode, mutant, ph, temperature, ddg, full_name, name, dist,
                #          atom_coord[0], atom_coord[1], atom_coord[2]])
                #     temp_array = temp_array.reshape(1, len(temp_array))
                #     temp_df = pd.DataFrame(temp_array)
                #     temp_df.columns = df_init.columns
                #     df_init = pd.concat([df_init, temp_df], axis=0, ignore_index=True)
                if dist <= radius:
                    temp_array = np.array(
                        [key, pdb, wild_type, chain, position, inode, rsa, mutant, ph, temperature, ddg, full_name, name, dist,
                         atom_coord[0], atom_coord[1], atom_coord[2]])
                    temp_array = temp_array.reshape(1, len(temp_array))
                    temp_df = pd.DataFrame(temp_array)
                    temp_df.columns = df_init.columns
                    df_init = pd.concat([df_init, temp_df], axis=0, ignore_index=True)
        # ==================== 设置列的数据格式 ====================
        #print('设置列格式前:\n',df_init.dtypes)
        df_init[['key']] = df_init[['key']].astype(int)
        df_init[['position']] = df_init[['position']].astype(int)
        df_init[['ph']] = df_init[['ph']].astype(float)
        df_init[['temperature']] = df_init[['temperature']].astype(float)
        df_init[['ddg']] = df_init[['ddg']].astype(float)
        df_init[['dist']] = df_init[['dist']].astype(float)
        df_init[['x']] = df_init[['x']].astype(float)
        df_init[['y']] = df_init[['y']].astype(float)
        df_init[['z']] = df_init[['z']].astype(float)
        df_init[['rsa']] = df_init[['rsa']].astype(float)
        #print('设置列格式后：\n',df_init.dtypes)
        #print('before sort',df_init)
        df_init = df_init.sort_values(by='dist', ascending = True) # 把df_init按照距离中心的距离排序，以便后期建立cutoff图片
        #print('after_sort',df_init)

        # 选择中心的 k 近邻原子作为突变邻域环境
        if k_neighbor != 0:
            assert len(df_init) >= k_neighbor
            df_init = df_init.iloc[0:k_neighbor, :] # 如果是k近邻，取前k个.

        df_init_after = transform(df_init, center_coord, model = pca_model)# 旋转加平移
        df_neighbor = pd.concat([df_neighbor, df_init_after])
    return df_neighbor


if __name__ == '__main__':
    dataset_name, radius, k_neighbor = sys.argv[1:]
    radius = float(radius)
    k_neighbor = int(k_neighbor)
    dataset_path = '../datasets/%s/'%dataset_name
    mutation_csv_path = dataset_path+'%s_new.csv'%dataset_name #描述突变信息的csv文件路径

    # 函数输入参数
    pdbfile_base_path = dataset_path + 'pdb' + dataset_name
    #k_neighbor = 30
    #radius = 10
    only_atom = True
    centerCA = True
    pca_model = 0

    coord_csv_path = dataset_path+'src_former_'+dataset_name+'_r_%.2f_neighbor_%d_onlyAtom_centerCA.csv'%(radius,k_neighbor)

    f = open(mutation_csv_path, 'r'); mutations_df = pd.read_csv(f); f.close()
    temp_df = mutations_df.drop_duplicates('PDB', 'first', inplace=False)#保存副本
    structure_name_list = list(temp_df.PDB)
    #print(structure_name_list)
    print('此数据集的文件名为：%s，共包含%d个突变信息，共发生在%d个蛋白质中.' % (dataset_name, len(mutations_df), len(structure_name_list)))
    del temp_df

    # 将所有的突变写入 csv文件, csv 结构如下：
    df_all = pd.DataFrame(
        {'key':[], 'pdb': [], 'wild_type': [], 'chain': [], 'position': [], 'inode':[], 'rsa':[], 'mutant': [], 'ph': [],
         'temperature': [], 'ddg': [], 'full_name': [], 'name': [], 'dist': [], 'x': [], 'y': [], 'z': []})

    #将每一个结构信息通过循环写入csv文件。
    for structure_name in structure_name_list:
        df_test = mutations_df[mutations_df.PDB == structure_name] # structure_name 包含突变位点和扩展名.
        df_neighbor = get_neighbor(df_test, radius, pdbfile_base_path, k_neighbor=k_neighbor, only_atom = only_atom, centerCA=centerCA, pca_model=pca_model)
        df_all = pd.concat([df_all, df_neighbor])
    print('df_all shape:',df_all.shape)# 17列
    df_all.to_csv(coord_csv_path, index=False)
