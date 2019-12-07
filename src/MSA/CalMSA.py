#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys
import numpy as np
from mCNN.processing import shell, aa_321dict, read_csv,PDBparser, check_qsub
from run_psiblast import qsub

def main():
    dataset_name, flag = sys.argv[1:] ## flag = ['W','R'] --> based on wild or refined structure
    HOMEdir = shell('echo $HOME')
    Calculator = CalculateMSA(dataset_name, HOMEdir)

    Calculator.pdb2seq(flag)
    Calculator.run_psiblast()
    Calculator.blast2msa()

class CalculateMSA(object):
    def __init__(self, dataset_name, homedir):
        self.aa_lst = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', '-']
        self.dataset_name = dataset_name
        self.homedir      = homedir
        self.appdir       = '/public/application/ncbi-blast-2.3.0+/bin/psiblast'
        self.alignblast   = '%s/mCNN/src/MSA/alignblast.pl'%homedir
        self.libdir       = '/library/nr/nr'

        self.pdb_path = '%s/mCNN/dataset/%s/pdb' %(homedir, dataset_name)
        self.ref_pdb_path  = '%s/mCNN/dataset/%s/feature/rosetta/ref_output' % (homedir, dataset_name)
        self.mut_pdb_path  = '%s/mCNN/dataset/%s/feature/rosetta/mut_output' % (homedir, dataset_name)
        self.wild_seq_path = '%s/mCNN/dataset/%s/feature/msa/wild' % (homedir, dataset_name)
        self.mut_seq_path  = '%s/mCNN/dataset/%s/feature/msa/ref' % (homedir, dataset_name)

        if not os.path.exists(self.wild_seq_path):
            os.makedirs(self.wild_seq_path)
        if not os.path.exists(self.mut_seq_path):
            os.makedirs(self.mut_seq_path)

    def pdb2seq(self,flag):
        tag_name_dict = {'wild': [],'mutant':[x for x in os.listdir(self.pdbpath)]}
        wild_pdb_name_lst  = []
        for tag_name in tag_name_dict['mutant']:
            pdbid = tag_name[:4]
            if not pdbid in wild_pdb_name_lst:
                wild_pdb_name_lst.append(pdbid)
                tag_name_dict['wild'].append(tag_name)
        del wild_pdb_name_lst

        for flag in ['wild','mutant']:
            for tag_name in tag_name_dict[flag]:
                pdbid = tag_name.split('_')[0]
                if flag == 'wild':
                    pdbdir = '%s/%s/%s_ref.pdb'%(self.pdbpath,tag_name,pdbid)
                else:
                    pdbdir = '%s/%s/%s_mut.pdb'%(self.pdbpath,tag_name,pdbid)
                print(pdbdir)
                model = PDBparser(pdbdir,MDL=0,write=0,outpath=None)
                for chain in model:
                    aalst=[]
                    chain_name = chain.get_id()
                    seq_outpath = '%s/%s/%s/%s'%(self.seq_outpath,flag,pdbid,chain_name)
                    if not os.path.exists(seq_outpath):
                        os.makedirs(seq_outpath)
                    for residue in chain:
                        aalst.append(aa_321dict[residue.get_resname()])
                    seq_outdir = '%s/%s_%s.fasta' % (seq_outpath, pdbid, chain_name)
                    g = open(seq_outdir, 'w')
                    g.writelines('>%s_%s.fasta\n'% (pdbid,chain_name))
                    g.writelines(''.join(aa for aa in aalst))
                    g.writelines('\n')
                    g.close()
        print('-'*10,'pdb to seq done!')

    def run_psiblast(self):
        qsub(self.appdir, self.libdir, self.seq_outpath)
        check_qsub()

    def blast2msa(self):
        for flag in ['wild', 'mutant']:
            flag_path = '%s/%s' % (self.seq_outpath, flag)
            for pdbid in os.listdir(flag_path):
                pdbid_path = '%s/%s' % (flag_path, pdbid)
                for chain_name in os.listdir(pdbid_path):
                    chain_path = '%s/%s' % (pdbid_path, chain_name)
                    seq = '%s/%s_%s.fasta' % (chain_path,pdbid,chain_name)
                    blast_out = '%s/blast.out' % chain_path
                    msa_aln = '%s/msa.aln' % chain_path
                    os.system('%s %s %s -Q %s -psi' % (self.alignblast, blast_out, msa_aln, seq))

                    self.msa2count(blastpath=chain_path)

        print('-'*10,'blast to count done')

    def msa2count(self,blastpath):
        msa = '%s/msa.aln' %blastpath
        msa_count = '%s/msa.count' %blastpath
        msa_freq = '%s/msa.freq' %blastpath
        msa_cnt_frq = '%s/msa.cnt_frq.npz' %blastpath

        f = open(msa, 'r')
        lines = f.readlines()
        f.close()
        seq = lines[0].strip('\n')
        seqlen = len(seq)

        g = open(msa_count, 'w')
        g.writelines('0 res ' + ' '.join(self.aa_lst) + '\n')
        for col_num in range(seqlen):
            col_str = ''.join([s[col_num].strip('\n') for s in lines])
            aa_count = ' '.join([str(col_str.count(aa)) for aa in self.aa_lst])
            g.writelines('%d %s %s\n' % (col_num + 1, seq[col_num], aa_count))
        g.close()

        countarr = np.loadtxt(msa_count, dtype=str)
        sub_countarr = countarr[1:, 2:].astype(float)
        sub_freqarr = (sub_countarr.T / np.sum(sub_countarr, axis=1)).T
        sub_freqarr_str = sub_freqarr.astype(str)
        freqarr = countarr.copy()
        freqarr = freqarr.astype(object)
        freqarr[1:, 2:] = sub_freqarr_str
        np.savetxt(msa_freq, freqarr, fmt='%s', delimiter=' ')
        np.savez(msa_cnt_frq, cnt=countarr, frq=freqarr, cnt_sub=sub_countarr, frq_sub=sub_freqarr)

if __name__ == '__main__':
    main()
