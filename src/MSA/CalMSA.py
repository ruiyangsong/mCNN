#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys, time, warnings
import numpy as np
from mCNN.processing import shell, aa_321dict, read_csv,PDBparser, check_qsub, log
from run_psiblast import qsub_wild, qsub_mutant

def main():
    dataset_name = sys.argv[1]
    HOMEdir = shell('echo $HOME')
    Calculator = CalculateMSA(dataset_name, HOMEdir)

    Calculator.pdb2seq()
    Calculator.run_psiblast()
    Calculator.blast2msa()


def qsub_wild(app, libdir, seq_path, qsubid_prefix):
    pdbid_dir_lst = [seq_path+'/'+x for x in os.listdir(seq_path)]
    for pdbdir in pdbid_dir_lst:
        pdbid = pdbdir.split('/')[-1][0:4]
        for chain_name in os.listdir(pdbdir):
            outdir = '%s/%s'%(pdbdir, chain_name)
            qsubid = '%s_%s_%s'%(qsubid_prefix,pdbid,chain_name)
            file = '%s/%s_%s.fasta'%(outdir,pdbid,chain_name)
            if not os.path.exists('%s/qsub_log'%outdir):
                os.makedirs('%s/qsub_log'%outdir)
            os.chdir(outdir)
            walltime = 'walltime=240:00:00'
            errfile = './qsublog/err'
            outfile = './qsublog/out'
            run_psiblast = 'run_prog.sh'
            blast_out = '%s/blast.out'%outdir
            os.system('touch %s'%blast_out)
            g = open(run_psiblast, 'w')
            g.writelines('#!/usr/bin/bash\n')
            g.writelines('blast_out=\"%s\"\n'%blast_out)
            g.writelines('echo $blast_out\n')
            g.writelines("echo 'user:' `whoami`\necho 'hostname:' `hostname`\necho 'begin at:' `date`\n")
            g.writelines('%s -query %s -db %s -out $blast_out -num_iterations 3\n'%(app,file,libdir))
            g.writelines("echo 'end at:' `date`\n")
            g.close()

            os.system('chmod 755 %s' % run_psiblast)
            os.system('/public/home/sry/bin/getQ.pl')
            os.system('qsub -e %s -o %s -l %s -N %s %s' % (errfile, outfile, walltime, qsubid, run_psiblast))
            time.sleep(1)

def qsub_mutant(app, libdir, seq_path, qsubid_prefix):
    tag_dir_lst = [seq_path+'/'+x for x in os.listdir(seq_path)]
    for tagdir in tag_dir_lst:
        tag = tagdir.split('/')[-1]
        pdbid,wtaa,chain_name = tag.split('_')[:3]
        outdir = tagdir
        qsubid = '%s_%s'%(qsubid_prefix,tag)
        file = '%s/%s_%s.fasta'%(outdir,pdbid,chain_name)
        if not os.path.exists('%s/qsub_log'%outdir):
            os.makedirs('%s/qsub_log'%outdir)
        os.chdir(outdir)
        walltime = 'walltime=240:00:00'
        errfile = './qsublog/err'
        outfile = './qsublog/out'
        run_psiblast = 'run_prog.sh'
        blast_out = '%s/blast.out'%outdir
        os.system('touch %s'%blast_out)
        g = open(run_psiblast, 'w')
        g.writelines('#!/usr/bin/bash\n')
        g.writelines('blast_out=\"%s\"\n'%blast_out)
        g.writelines('echo $blast_out\n')
        g.writelines("echo 'user:' `whoami`\necho 'hostname:' `hostname`\necho 'begin at:' `date`\n")
        g.writelines('%s -query %s -db %s -out $blast_out -num_iterations 3\n'%(app,file,libdir))
        g.writelines("echo 'end at:' `date`\n")
        g.close()

        os.system('chmod 755 %s' % run_psiblast)
        os.system('/public/home/sry/bin/getQ.pl')
        os.system('qsub -e %s -o %s -l %s -N %s %s' % (errfile, outfile, walltime, qsubid, run_psiblast))
        time.sleep(1)


class CalculateMSA(object):
    def __init__(self, dataset_name, homedir):
        self.aa_lst = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', '-']
        self.dataset_name = dataset_name
        self.homedir      = homedir
        self.sleep_time   = 5
        self.appdir       = '/public/application/ncbi-blast-2.3.0+/bin/psiblast'
        self.alignblast   = '%s/mCNN/src/MSA/alignblast.pl'%homedir
        self.libdir       = '/library/nr/nr'
        self.mt_csv_dir   = '%s/mCNN/dataset/%s/%s.csv' %(homedir, dataset_name,dataset_name)

        self.mdl0_pdb_path  = '%s/mCNN/dataset/%s/feature/rosetta/global_output/pdb_mdl0' %(homedir, dataset_name)# mdl pdb

        self.wild_seq_path    = '%s/mCNN/dataset/%s/feature/msa/mdl0_wild' % (homedir, dataset_name)  # native wild sequence
        self.mutant_seq_path  = '%s/mCNN/dataset/%s/feature/msa/mdl0_mutant' % (homedir, dataset_name)  # native mutant sequence

        if not os.path.exists(self.wild_seq_path):
            os.makedirs(self.wild_seq_path)
        if not os.path.exists(self.mutant_seq_path):
            os.makedirs(self.mutant_seq_path)

    @log
    def pdb2seq(self):
        '''based on native mdl0 pdbs, these pdbs may slightly different on some residues compared to rosetta refined pdbs'''
        df = read_csv(self.mt_csv_dir)
        pdbdir_lst = [self.mdl0_pdb_path+'/'+x for x in os.listdir(self.mdl0_pdb_path)]
        for pdbdir in pdbdir_lst:
            pdbid = pdbdir.split('/')[-1][0:4]
            model = PDBparser(pdbdir,MDL=0,write=0,outpath=None)

            ## only get mutant sequence for each pdbid
            df_pdb = df.loc[df.PDB == pdbid, :]
            for i in range(len(df_pdb)):
                WILD_TYPE, CHAIN, POSITION, MUTANT = df_pdb.iloc[i, 2:6]
                chain = model[CHAIN]
                # @@++ #################################################################################################
                # mdl_0生成时只考虑了标准残基，但标准残基可能是HETATM, ie., res.get_id()[0] != ' ',故去除了 str(res.get_id()[0])
                # res_id_lst = [(str(res.get_id()[0]) + str(res.get_id()[1]) + str(res.get_id()[2])).strip() for res in chain]# @@++
                # @@++ #################################################################################################
                res_id_lst = [(str(res.get_id()[1]) + str(res.get_id()[2])).strip() for res in chain]
                res_name_lst = [aa_321dict[res.get_resname()] for res in chain]
                index = res_id_lst.index(str(POSITION))
                assert res_name_lst[index] == WILD_TYPE
                res_name_lst[index] = MUTANT

                if len(res_name_lst) > 0:
                    seq_outpath = '%s/%s_%s_%s_%s_%s' % (self.mutant_seq_path, pdbid, WILD_TYPE,CHAIN,POSITION,MUTANT)
                    if not os.path.exists(seq_outpath):
                        os.makedirs(seq_outpath)
                    seq_outdir = '%s/%s_%s.fasta' % (seq_outpath, pdbid, CHAIN)
                    g = open(seq_outdir, 'w')
                    g.writelines('>%s_%s_mdl_0.fasta,date:%s,%s\n' % (pdbid, CHAIN, time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()),','.join(res_id_lst)))
                    g.writelines(''.join(aa for aa in res_name_lst))
                    g.writelines('\n')
                    g.close()
                else:
                    warnings.warn('WARNING: Chain %s do not contains target atoms.' % CHAIN)

            # get all sequences for wild pdb
            for chain in model:
                chain_name   = chain.get_id()
                # @@++ #################################################################################################
                # mdl_0生成时只考虑了标准残基，但标准残基可能是HETATM, ie., res.get_id()[0] != ' ',故去除了 str(res.get_id()[0])
                # res_id_lst = [(str(res.get_id()[0]) + str(res.get_id()[1]) + str(res.get_id()[2])).strip() for res in chain]# @@++
                # @@++ #################################################################################################
                res_id_lst   = [(str(res.get_id()[1])+str(res.get_id()[2])).strip() for res in chain]
                res_name_lst = [aa_321dict[res.get_resname()] for res in chain]
                assert len(res_id_lst) == len(res_name_lst)
                if len(res_name_lst) > 0:
                    seq_outpath = '%s/%s/%s' % (self.wild_seq_path, pdbid, chain_name)
                    if not os.path.exists(seq_outpath):
                        os.makedirs(seq_outpath)
                    seq_outdir = '%s/%s_%s.fasta' % (seq_outpath, pdbid, chain_name)
                    g = open(seq_outdir, 'w')
                    g.writelines('>%s_%s_mdl_0.fasta,date:%s,%s\n' % (pdbid, chain_name, time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()),','.join(res_id_lst)))
                    g.writelines(''.join(aa for aa in res_name_lst))
                    g.writelines('\n')
                    g.close()
                else:
                    warnings.warn('WARNING: Chain %s do not contains target atoms.' % chain_name)
        print('---pdb2seq done!')

    @log
    def run_psiblast(self):
        wild_tag   = 'psiblast_%s_wt'%self.dataset_name
        mutant_tag = 'psiblast_%s_mt'%self.dataset_name
        blast_tag  = 'psiblast_%s'%self.dataset_name
        qsub_wild(app=self.appdir, libdir=self.libdir, seq_path=self.wild_seq_path, qsubid_prefix = wild_tag)
        qsub_mutant(app=self.appdir, libdir=self.libdir, seq_path=self.mutant_seq_path, qsubid_prefix=mutant_tag)
        check_qsub(tag=blast_tag,sleep_time = self.sleep_time)

    @log
    def blast2msa(self):
        wild_err_count = 0
        wild_pdbid_dir_lst = [self.wild_seq_path + '/' + x for x in os.listdir(self.wild_seq_path)]
        for pdbdir in wild_pdbid_dir_lst:
            pdbid = pdbdir.split('/')[-1][0:4]
            for chain_name in os.listdir(pdbdir):
                chain_dir = '%s/%s' % (pdbdir, chain_name)
                seq = '%s/%s_%s.fasta' % (chain_dir,pdbid,chain_name)
                blast_out = '%s/blast.out' % chain_dir
                msa_aln = '%s/msa.aln' % chain_dir
                try:
                    os.system('%s %s %s -Q %s -psi' % (self.alignblast, blast_out, msa_aln, seq))
                    self.msa2count(blast_dir=chain_dir,seqdir = seq)
                except:
                    wild_err_count += 1
                    print('\n[ERROR] ERROE dir of blast2msa: %s' % chain_dir)
        if wild_err_count > 0:
            print('!!!ERROE of blast2msa (wild) occurs %s time(s)!!!' % wild_err_count)

        mutant_err_count = 0
        mutant_pdbid_dir_lst = [self.mutant_seq_path + '/' + x for x in os.listdir(self.mutant_seq_path)]
        for tagdir in mutant_pdbid_dir_lst:
            tag = tagdir.split('/')[-1]
            pdbid, wtaa, chain_name = tag.split('_')[:3]
            seq = '%s/%s_%s.fasta' % (tagdir, pdbid, chain_name)
            blast_out = '%s/blast.out' % tagdir
            msa_aln = '%s/msa.aln' % tagdir
            try:
                os.system('%s %s %s -Q %s -psi' % (self.alignblast, blast_out, msa_aln, seq))
                self.msa2count(blast_dir=tagdir, seqdir=seq)
            except:
                mutant_err_count += 1
                print('\n[ERROR] ERROE dir of blast2msa: %s' % tagdir)
        if mutant_err_count > 0:
            print('!!!ERROE of blast2msa (mutant) occurs %s time(s)!!!' % wild_err_count)

        print('---blast to count done!')

    def msa2count(self,blast_dir,seqdir):
        msa = '%s/msa.aln' %blast_dir
        msa_count = '%s/msa.count' %blast_dir
        msa_freq = '%s/msa.freq' %blast_dir
        msa_cnt_frq = '%s/msa.cnt_frq.npz' %blast_dir

        sf = open(seqdir, 'r')
        sflines = sf.readlines()
        sf.close()
        indexlst = sflines[0].strip().split(',')[2:]

        f = open(msa, 'r')
        lines = f.readlines()
        f.close()

        seq = lines[0].strip('\n')
        seqlen = len(seq)
        g = open(msa_count, 'w')
        g.writelines('0 pos res ' + ' '.join(self.aa_lst) + '\n')

        for col_num in range(seqlen):
            col_str = ''.join([s[col_num].strip('\n') for s in lines])
            aa_count = ' '.join([str(col_str.count(aa)) for aa in self.aa_lst])
            g.writelines('%d %s %s %s\n' % (col_num + 1, indexlst[col_num], seq[col_num], aa_count))
        g.close()

        countarr = np.loadtxt(msa_count, dtype=str)
        sub_countarr = countarr[1:, 3:].astype(float)
        sub_freqarr = (sub_countarr.T / np.sum(sub_countarr, axis=1)).T
        sub_freqarr_str = sub_freqarr.astype(str)
        freqarr = countarr.copy()
        freqarr = freqarr.astype(object)
        freqarr[1:, 3:] = sub_freqarr_str
        np.savetxt(msa_freq, freqarr, fmt='%s', delimiter=' ')
        np.savez(msa_cnt_frq, cnt=countarr, frq=freqarr, cnt_sub=sub_countarr, frq_sub=sub_freqarr)

if __name__ == '__main__':
    main()
