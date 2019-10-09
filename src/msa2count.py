#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os,sys
import numpy as np
# import pandas as pd
dataset_name = sys.argv[1]
# dataset_name = 'S2648'
aa_lst = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L',
          'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', '-']
datadir = '../msa/psiblast%s'%dataset_name #msa/psiblastS2648
blastname_lst = [x for x in os.listdir(datadir) if os.path.isdir('%s/%s'%(datadir,x))]
for blastname in blastname_lst:
    seq = '%s/%s/%s.fasta'%(datadir,blastname,blastname[9:])
    msa = '%s/%s/msa.aln'%(datadir,blastname)
    msa_count = '%s/%s/msa.count'%(datadir,blastname)
    msa_freq = '%s/%s/msa.freq' % (datadir, blastname)
    msa_cnt_frq = '%s/%s/msa.cnt_frq.npz' % (datadir, blastname)
    sf = open(seq,'r')
    sflines = sf.readlines()
    sf.close()
    indexlst = sflines[0].strip('>').split(',')[:-1]
    seqorigin = sflines[1].strip('\n')

    f = open(msa,'r')
    lines = f.readlines()
    f.close()

    g = open(msa_count,'w+')
    g.writelines('0 pos res '+' '.join(aa_lst)+'\n')
    seq = lines[0].strip('\n')
    #print(indexlst)
    assert seq == seqorigin
    seqlen = len(seq)
    assert len(indexlst) == seqlen
    # print('seq len: %s'%seqlen)
    # print('total number of seq in msa: %s'%len(lines))
    for col_num in range(seqlen):
        col_str = ''.join([s[col_num].strip('\n') for s in lines])
        aa_count = ' '.join([str(col_str.count(aa)) for aa in aa_lst])
        g.writelines('%d %s %s %s\n'%(col_num+1, indexlst[col_num], seq[col_num], aa_count))
        # g.writelines('%d %s %s\n' % (col_num + 1, seq[col_num], aa_count))
    g.close()

    # df = pd.read_table(msa_count,delim_whitespace=True,index_col=0)
    # df.to_csv('%s/%s/msa.csv'%(datadir, blastname))
    countarr = np.loadtxt(msa_count,dtype=str)
    # print(countarr)
    sub_countarr = countarr[1:,3:].astype(float)
    sub_freqarr = (sub_countarr.T/np.sum(sub_countarr,axis=1)).T
    sub_freqarr_str = sub_freqarr.astype(str)
    # print(sub_freqarr_str)
    freqarr = countarr.copy()
    freqarr = freqarr.astype(object)
    freqarr[1:, 3:] = sub_freqarr_str
    # freqarr = freqarr.astype(str)
    # print(freqarr)
    np.savetxt(msa_freq,freqarr,fmt='%s',delimiter=' ')
    np.savez(msa_cnt_frq, cnt = countarr, frq = freqarr, cnt_sub =sub_countarr, frq_sub = sub_freqarr)
