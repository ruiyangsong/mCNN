#!/usr/bin/env python
# -*- coding: utf-8 -*-
import re
import os
import time

#gfs means get_from_shell
def gfs(cmd):
	res=os.popen(cmd).readlines()[0].strip()
	return res

node=gfs('hostname')
datetime=gfs('date')
path=gfs('pwd')
print "Hostname: %s\n" %node
print "Started : %s\n" %datetime
print "Path    : %s\n" %path

libdir="!LIBDIR!"
user="!USER!"
tag="!TAG!"
outdir="!OUTDIR!"
name="!NAME!"
outblast="!OUTBLAST!"

workdir="/tmp/"+user+"/"+tag
os.system("mkdir -p "+workdir)
os.system("rm -rf "+workdir+"/*")
os.chdir(workdir)

datadir="!DATADIR!"
filename="!FILENAME!"

os.system("cp %s ." %(datadir+'/'+name+'/'+filename))
os.system("/public/home/caobx/local/app/blast/bin/psiblast -query "+filename+" -db "+libdir+" -out blast.out -num_iterations 3")
os.system("mv -f ./blast.out "+outdir+'/'+outblast)

tim=gfs('date')
print "Ending time: %s" %tim
time.sleep(1)

exit()
