#!/usr/bin/env python
# -*- coding: utf-8 -*-
import re
import os
import time

def shell(cmd):
	res=os.popen(cmd).readlines()[0].strip()
	return res

node=shell('hostname')
begintime=shell('date')
path=shell('pwd')

print('Hostname: %s\n' %node)
print('Started : %s\n' %begintime)
print('Path    : %s\n' %path)

#----------BEGIN----------
appdir = '!APPDIR!'
libdir = '!LIBDIR!'
datadir = '!DATADIR!'
filename='!FILENAME!' #each seq name.

user='!USER!'
tag='!TAG!'
outdir='!OUTDIR!'
#name='!NAME!'
outblast='!OUTBLAST!'

workdir='/tmp/'+user+'/'+tag
os.system('mkdir -p ' + workdir)
os.system('rm -rf ' + workdir + '/*')
os.chdir(workdir)

os.system('cp %s ./' %(datadir + '/' +filename))
os.system(appdir + ' -query ' + filename + ' -db ' + libdir + ' -out blast.out -num_iterations 3')
os.system('mv -f ./blast.out ' + outdir)

#----------END----------
endtime = shell('date')
print('Ending time: %s' % endtime)
time.sleep(1)

exit()
