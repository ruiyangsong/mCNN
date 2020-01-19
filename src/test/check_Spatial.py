#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys, os
from mCNN.processing import shell

dataset_name = sys.argv[1]
homedir    = shell('echo $HOME')

def check_mCNN(homedir,dataset_name):
    wild_qsublog_dir = '%s/mCNN/dataset/%s/feature/mCNN/wild/qsublog'%(homedir,dataset_name)
    mutant_qsublog_dir = '%s/mCNN/dataset/%s/feature/mCNN/mutant/qsublog'%(homedir,dataset_name)

    wild_qsubtag_dir_lst = [wild_qsublog_dir+'/'+x for x in os.listdir(wild_qsublog_dir)]
    mutant_qsubtag_dir_lst = [mutant_qsublog_dir+'/'+x for x in os.listdir(mutant_qsublog_dir)]

    for wild_qsubtag_dir in wild_qsubtag_dir_lst:
        errdirlst = [wild_qsubtag_dir+'/'+x+'/err' for x in os.listdir(wild_qsubtag_dir)]
        for errdir in errdirlst:
            if not os.path.exists(errdir):
                print('DO Not Exist',errdir)
                continue
            if int(os.path.getsize(errdir)) > 0:
                # print(os.path.getsize(errdir))
                # print(type(os.path.getsize(errdir)))
                print(errdir)

    for mutant_qsubtag_dir in mutant_qsubtag_dir_lst:
        errdirlst = [mutant_qsubtag_dir+'/'+x+'/err' for x in os.listdir(mutant_qsubtag_dir)]
        for errdir in errdirlst:
            if not os.path.exists(errdir):
                print('DO Not Exist',errdir)
                continue
            if int(os.path.getsize(errdir)) > 0:
                # print(os.path.getsize(errdir))
                # print(type(os.path.getsize(errdir)))
                print(errdir)

def check_mCSM(homedir,dataset_name):
    wild_qsublog_dir = '%s/mCNN/dataset/%s/feature/mCSM/wild/qsublog' % (homedir,dataset_name)
    mutant_qsublog_dir = '%s/mCNN/dataset/%s/feature/mCSM/mutant/qsublog' %(homedir, dataset_name)

    wild_qsubtag_dir_lst = [wild_qsublog_dir + '/' + x +'/err' for x in os.listdir(wild_qsublog_dir)]
    mutant_qsubtag_dir_lst = [mutant_qsublog_dir + '/' + x+'/err' for x in os.listdir(mutant_qsublog_dir)]

    for errdir in wild_qsubtag_dir_lst:
        if not os.path.exists(errdir):
            print('DO Not Exist',errdir)
            continue
        if int(os.path.getsize(errdir)) > 0:
            # print(os.path.getsize(errdir))
            # print(type(os.path.getsize(errdir)))
            print(errdir)


    for errdir in mutant_qsubtag_dir_lst:
        if not os.path.exists(errdir):
            print('DO Not Exist',errdir)
            continue
        if int(os.path.getsize(errdir)) > 0:
            # print(os.path.getsize(errdir))
            # print(type(os.path.getsize(errdir)))
            print(errdir)


if __name__ == '__main__':
    check_mCNN(homedir,dataset_name)
    check_mCSM(homedir,dataset_name)