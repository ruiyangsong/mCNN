#!/usr/bin/env python
import os,json
import numpy as np
import pandas as pd

def main():
    csvpth = '/public/home/sry/mCNN/dataset/TR/S2648_TR500.csv'
    df = pd.read_csv(csvpth)
    for i in range(len(df)):
        columns = ['key', 'PDB', 'WILD_TYPE', 'CHAIN', 'POSITION', 'MUTANT', 'PH', 'TEMPERATURE', 'DDG']
        key, PDB, WILD_TYPE, CHAIN, POSITION, MUTANT, PH, TEMPERATURE, DDG = df.iloc[i,:][columns].values
        ## for wild
        wild_pdb_pth = '/public/home/sry/mCNN/dataset/TR/pdb_chain/%s.pdb'%PDB
        mutant_tag = '%s.%s.%s.%s.%s.%s'%(key,PDB,WILD_TYPE,CHAIN,POSITION,MUTANT)
        mutant_pdb_pth = '/public/home/sry/mCNN/dataset/TR/output/%s/model1.pdb'%(mutant_tag) #2437.3SIL.A.A.53.L
        outdir = '/public/home/sry/mCNN/dataset/TR/feature/TMalign/wild_TR_%s'%mutant_tag
        res_dict = TMalign(pdb1pth=wild_pdb_pth,pdb2pth=mutant_pdb_pth,outdir=outdir)

        ## for TR output
        wild_tag = '%s.%s.%s'%(key,PDB,CHAIN)
        wild_pdb_pth = '/public/home/sry/mCNN/dataset/TR/output/%s/model1.pdb'%(wild_tag)
        mutant_tag = '%s.%s.%s.%s.%s.%s' % (key, PDB, WILD_TYPE, CHAIN, POSITION, MUTANT)
        mutant_pdb_pth = '/public/home/sry/mCNN/dataset/TR/output/%s/model1.pdb' % (mutant_tag)  # 2437.3SIL.A.A.53.L
        outdir = '/public/home/sry/mCNN/dataset/TR/feature/TMalign/TR_TR_%s' % mutant_tag
        res_dict = TMalign(pdb1pth=wild_pdb_pth, pdb2pth=mutant_pdb_pth, outdir=outdir)

def TMalign(pdb1pth,pdb2pth,outdir='.'):
    os.makedirs(outdir,exist_ok=True)
    # res = os.popen('TMalign %s %s -m mat.txt'%(pdb1pth,pdb2pth))
    res = os.popen('TMalign %s %s'%(pdb1pth,pdb2pth))
    idx = 4
    reslst = []
    mat = []
    for row in res.readlines():
        row = row.strip()
        if row.strip().find('RMSD')!=-1:
            # print(row)
            reslst.append(row)
        if row.strip().find('TM-score=')!=-1:
            reslst.append(row)
        if row.strip().find('Rotation matrix')!=-1:
            idx=0
            continue
        if idx <= 3:
            # print(row)
            reslst.append(row)
            idx+=1

    # for x in reslst:
    #     print(x)
    # print('-' * 100)
    RMSD = float(reslst[0].split(',')[1].split('=')[-1].strip())
    Tmscore1 = float(reslst[1].split('(')[0].split('=')[-1].strip())
    Tmscore2 = float(reslst[2].split('(')[0].split('=')[-1].strip())
    mat.append([float(x) for x in reslst[4].split()[1:]])
    mat.append([float(x) for x in reslst[5].split()[1:]])
    mat.append([float(x) for x in reslst[6].split()[1:]])
    res_dict = {'RMSD':RMSD,'Tmscore1':Tmscore1,'Tmscore2':Tmscore2,'mat':mat}

    with open('%s/res_dict.json'%outdir, 'w') as f:
        f.write(json.dumps(res_dict))
    return res_dict

if __name__ == '__main__':
    main()