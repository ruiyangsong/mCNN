#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys
from mCNN.processing import shell, log

def main():
    dataset_name = sys.argv[1]
    homedir = shell('echo $HOME')
    appdir = '%s/mCNN/src/Stride/stride' %homedir
    pdb_wild_dir = '%s/mCNN/dataset/%s/pdb_chain'%(homedir, dataset_name)#wild_pdb_dir
    pdb_TR_dir = '%s/mCNN/dataset/%s/output'%(homedir, dataset_name)
    outdir_wild = '%s/mCNN/dataset/%s/feature/stride/wild'%(homedir, dataset_name)
    outdir_TR   = '%s/mCNN/dataset/%s/feature/stride/TR'%(homedir, dataset_name)
    calSA(appdir,outdir_wild,outdir_TR,pdb_wild_dir,pdb_TR_dir)


@log
def calSA(appdir,outdir_wild,outdir_TR,pdb_wild_dir,pdb_TR_dir):

    if not os.path.exists(outdir_wild):
        os.makedirs(outdir_wild)
    if not os.path.exists(outdir_TR):
        os.makedirs(outdir_TR)

    pdb_lst = [x for x in os.listdir(pdb_wild_dir)]
    TR_tag_lst = [x for x in os.listdir(pdb_TR_dir)]

    for pdb in pdb_lst:
        pdbid = pdb[:4]
        pdbdir = '%s/%s.pdb'%(pdb_wild_dir,pdbid)
        os.system('%s %s > %s/%s.stride' % (appdir, pdbdir, outdir_wild, pdbid))

    for tagname in TR_tag_lst:
        pdbdir = '%s/%s/model1.pdb'%(pdb_TR_dir,tagname)
        os.system('%s %s > %s/%s.stride' % (appdir, pdbdir, outdir_TR, tagname))
    print('---stride done!')
if __name__ == '__main__':
    main()