#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys
from mCNN.processing import shell, log

def main():
    dataset_name = sys.argv[1]
    homedir = shell('echo $HOME')
    appdir = '%s/mCNN/src/Stride/stride' %homedir
    pdb_ref_path = '%s/mCNN/dataset/%s/feature/rosetta/ref_output'%(homedir, dataset_name)
    pdb_mut_path = '%s/mCNN/dataset/%s/feature/rosetta/mut_output'%(homedir, dataset_name)
    outdir_mutant = '%s/mCNN/dataset/%s/feature/stride/mutant'%(homedir, dataset_name)
    outdir_wild   = '%s/mCNN/dataset/%s/feature/stride/wild'%(homedir, dataset_name)
    calSA(appdir,outdir_mutant,outdir_wild,pdb_ref_path,pdb_mut_path)

@log
def calSA(appdir,outdir_mutant,outdir_wild,pdb_ref_path,pdb_mut_path):

    if not os.path.exists(outdir_mutant):
        os.makedirs(outdir_mutant)
    if not os.path.exists(outdir_wild):
        os.makedirs(outdir_wild)

    ref_tag_name_lst = [x for x in os.listdir(pdb_ref_path)]
    mut_tag_name_lst = [x for x in os.listdir(pdb_mut_path)]

    for pdbid in ref_tag_name_lst:
        pdbdir = '%s/%s/%s_ref.pdb'%(pdb_ref_path,pdbid,pdbid)
        os.system('%s %s > %s/%s.stride' % (appdir, pdbdir, outdir_wild, pdbid))

    for tagname in mut_tag_name_lst:
        pdbid = tagname.split('_')[0]
        pdbdir = '%s/%s/%s_mut.pdb'%(pdb_mut_path,tagname,pdbid)
        os.system('%s %s > %s/%s.stride' % (appdir, pdbdir, outdir_mutant, tagname))
    print('---stride done!')
if __name__ == '__main__':
    main()