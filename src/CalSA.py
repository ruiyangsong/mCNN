#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
** cal SA for each pdb file from program stride or DSSP.
'''
import os, argparse
#
# ASA_dict = {'A': 110.2, 'C': 140.4, 'D': 144.1, 'E': 174.7, 'F': 200.7,
#             'G': 78.7,  'H': 181.9, 'I': 185.0, 'K': 205.7, 'L': 183.1,
#             'M': 200.1, 'N': 146.4, 'P': 141.9, 'Q': 178.6, 'R': 229.0,
#             'S': 117.2, 'T': 138.7, 'V': 153.7, 'W': 240.5, 'Y': 213.7}

parser = argparse.ArgumentParser()
parser.add_argument('dataset_name', type=str, help='dataset name')
parser.add_argument('-A', '--APP', type=str, choices=['stride','DSSP'], default='stride', help='The calc program, stride or DSSP')
args = parser.parse_args()
dataset_name = args.dataset_name
if args.APP:
    APP = args.APP
PDBPATH = '../datasets/%s/pdb%s'%(dataset_name, dataset_name)
OUTDiR = '../datasets/%s/%s%s'%(dataset_name, APP, dataset_name)
if not os.path.exists(OUTDiR):
        os.mkdir(OUTDiR)
pdbdirlst = [PDBPATH + '/' + x for x in os.listdir(PDBPATH)]

for pdbdir in pdbdirlst:
    pdbname = pdbdir.split('/')[-1][0:4]
    # print('%s %s > %s/%s.%s'%(APP,pdbdir,OUTDiR,pdbname,APP))
    os.system('%s %s > %s/%s.%s'%(APP,pdbdir,OUTDiR,pdbname,APP))

print('-'*10, 'Calculate ASA done!')

if __name__ == '__main__':
    pass