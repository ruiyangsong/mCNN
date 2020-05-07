#!/usr/bin/env python
'''chk qsub results err'''
import os
wild_err_lst = []
mutant_err_lst = []
k_lst = [30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200]
wild_base_dir = '/public/home/sry/mCNN/dataset/SSD/feature/mCNN/wild/csv'
mutant_base_dir = '/public/home/sry/mCNN/dataset/SSD/feature/mCNN/mutant/csv'

wild_dir_lst = [wild_base_dir+'/'+x for x in os.listdir(wild_base_dir)]
mutant_dir_lst = [mutant_base_dir+'/'+x for x in os.listdir(mutant_base_dir)]

##for wild
for wild_dir in wild_dir_lst:
    for k in k_lst:
        try:
            assert os.path.exists('%s/center_CA_neighbor_%s.csv'%(wild_dir,k))
        except:
            wild_err_lst.append('%s/center_CA_neighbor_%s.csv'%(wild_dir,k))
print(wild_err_lst)

##for mutant
for mutant_dir in mutant_dir_lst:
    for k in k_lst:
        try:
            assert os.path.exists('%s/center_CA_neighbor_%s.csv'%(mutant_dir,k))
        except:
            mutant_err_lst.append('%s/center_CA_neighbor_%s.csv'%(mutant_dir,k))
print(mutant_err_lst)
