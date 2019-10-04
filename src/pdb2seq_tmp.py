#!/usr/local/bin/python
# -*- coding: utf-8 -*-
import os
def shell(cmd):
    res=os.popen(cmd).readlines()[0].strip()
    return res
node = shell('hostname')
begintime = shell('date')
path = shell('pwd')
print('--Hostname: %s\n' %node)
print('--Started : %s\n' %begintime)
print('--Path    : %s\n' %path)
# ---------- params ----------
user = '!USER!'
app = '!APP!'
funcdir = '!FUNCDIR!'
datadir = '!DATADIR!' #pdbS2648
filename = '!FILENAME!'
outdir = '!OUTDIR!'
tag = '!TAG!' # app_subid, eg: pdb2seq_S2648_WT0001
tmpdir = '!TMPDIR!' #/tmp/sry/pdb2seq_S2648_WT0001

seqname = '!SEQNAME!'
mdlid = 0
chainid = '!CHAINID!'
wtflag = '!WTFLAG!'
position = '!POSITION!'
mtaa = '!MTAA!'
# ---------- action ----------
os.system('mkdir -p %s/data %s/log %s/out'%(tmpdir,tmpdir,tmpdir))
os.system('rm -rf %s/data/* %s/log/* %s/out/*'%(tmpdir,tmpdir,tmpdir))
os.chdir(tmpdir)
os.system('cp %s/%s.pdb ./data'%(datadir,filename))
os.system('cp %s/pdb2seq.py .'%funcdir)
# os.system('%s pdb2seq.py >> ./log/%s.log'%(app,tag))
os.system('conda activate bio')
os.system('python pdb2seq.py %s %s %d %s %s %s %s'%(seqname, filename, mdlid, chainid, wtflag, position, mtaa))
os.system('mv -f %s %s'%(tmpdir, outdir))
#----------END----------
endtime = shell('date')
print('--Ending time: %s' % endtime)
exit()