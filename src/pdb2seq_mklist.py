#!/usr/local/bin/python
# -*- coding: utf-8 -*-
import sys, os, time
import pandas as pd
dataset_name = sys.argv[1]
path_csv_mutation = '../datasets/%s/%s_new_test.csv'%(dataset_name, dataset_name)
#---------
user = 'sry'
app = '/usr/local/bin/python'
funcdir = '/public/home/sry/mCNN/src'
datadir = '/public/home/sry/mCNN/datasets/%s/pdb%s'%(dataset_name, dataset_name) #pdbS2648

def main():
    f = open(path_csv_mutation, 'r')
    df_mutation = pd.read_csv(f)
    f.close()
    df_wt = df_mutation.loc[:,['PDB', 'CHAIN']]
    df_wt.drop_duplicates(keep='first', inplace=True)
    wt_num = len(df_wt)
    mut_num = len(df_mutation)
    for i in range(wt_num):
        wt_index = df_wt.iloc[i, :].values
        filename, chainid = wt_index[0][0:4] , wt_index[1]
        outdir = '/public/home/sry/mCNN/datasets/%s/seq%s'%(dataset_name, dataset_name)
        tag = 'pdb2seq_%s_WT_%s_%04d'%(dataset_name,filename,i+1)  # app_subid, eg: pdb2seq_S2648_WT0001
        tmpdir = '/tmp/%s/%s' % (user, tag)  # /tmp/sry/pdb2seq_S2648_WT0001
        seqname = '%s/WT_%s_%s_%04d'%(outdir,dataset_name,filename,i+1)
        wtflag = 'WT'
        position = ' '
        mtaa = '0'
        # print(filename, chainid, outdir, tag)
        # ---- construct based on tmp ----
        f = open('./pdb2seq_tmp.py', 'r')
        a = f.read()
        f.close()
        a = a.replace('!USER!', user)
        a = a.replace('!APP!', app)
        a = a.replace('!FUNCDIR!', funcdir)
        a = a.replace('!DATADIR!', datadir)
        a = a.replace('!FILENAME!', filename)
        a = a.replace('!OUTDIR!', outdir)
        a = a.replace('!TAG!', tag)
        a = a.replace('!TMPDIR!', tmpdir)

        a = a.replace('!SEQNAME!', seqname)
        a = a.replace('!WTFLAG!', wtflag)
        a = a.replace('!POSITION!', position)
        a = a.replace('!MTAA!', mtaa)

        if not os.path.exists(tmpdir):
            os.system('mkdir -p %s/qsub_log' % tmpdir)
        errfile = '%s/qsub_log/qsub_%s_err'% (tmpdir, tag)
        outfile = '%s/qsub_log/qsub_%s_out'% (tmpdir,tag)
        run_pdb2seq = '%s/run_pdb2seq_%s.py'%(tmpdir,tag)
        g = open(run_pdb2seq, 'w+')
        g.write(a)
        g.close()
        os.system('/public/home/sry/bin/getQ.pl')
        os.system('chmod 755 %s' % run_pdb2seq)
        walltime = 'walltime = 24:00:00'
        os.system('qsub -e %s -o %s -l %s -N %s %s' % (errfile, outfile, walltime, tag, run_pdb2seq))
        print('%s submitted\n' % run_pdb2seq)
        time.sleep(1)

    # =========================================================
    for i in range(mut_num):
        wtflag = 'MT'
        mt_index = df_mutation.loc[i, ['PDB','CHAIN','POSITION', 'MUTANT']].values
        filename, chainid, position, mtaa = mt_index[0][0:4], mt_index[1], mt_index[2], mt_index[3]
        outdir = '/public/home/sry/mCNN/datasets/%s/seq%s' % (dataset_name, dataset_name)
        tag = 'pdb2seq_%s_MT_%s_%04d' % (dataset_name, filename, i + 1)
        tmpdir = '/tmp/%s/%s' % (user, tag)
        seqname = '%s/MT_%s_%s_%04d' % (outdir, dataset_name, filename, i + 1)

        position = ' '
        mtaa = '0'
        # print(filename, chainid, outdir, tag)
        # ---- construct based on tmp ----
        f = open('./pdb2seq_tmp.py', 'r')
        a = f.read()
        f.close()
        a = a.replace('!USER!', user)
        a = a.replace('!APP!', app)
        a = a.replace('!FUNCDIR!', funcdir)
        a = a.replace('!DATADIR!', datadir)
        a = a.replace('!FILENAME!', filename)
        a = a.replace('!OUTDIR!', outdir)
        a = a.replace('!TAG!', tag)
        a = a.replace('!TMPDIR!', tmpdir)

        a = a.replace('!SEQNAME!', seqname)
        a = a.replace('!WTFLAG!', wtflag)
        a = a.replace('!POSITION!', position)
        a = a.replace('!MTAA!', mtaa)

        if not os.path.exists(tmpdir):
            os.system('mkdir -p %s/qsub_log' % tmpdir)
        errfile = '%s/qsub_log/qsub_%s_err' % (tmpdir, tag)
        outfile = '%s/qsub_log/qsub_%s_out' % (tmpdir,tag)
        run_pdb2seq = '%s/run_%s.py' % (tmpdir, tag)
        g = open(run_pdb2seq, 'w+')
        g.write(a)
        g.close()
        os.system('/public/home/sry/bin/getQ.pl')
        os.system('chmod 755 %s' % run_pdb2seq)
        walltime = 'walltime = 24:00:00'
        os.system('qsub -e %s -o %s -l %s -N %s %s' % (errfile, outfile, walltime, tag, run_pdb2seq))
        print('%s submitted\n' % run_pdb2seq)
        time.sleep(1)
if __name__ == '__main__':
    main()