#!/usr/bin/env python
# -*- coding: utf-8 -*-
import re
import os
import time

libdir="/public/library/uniref100/uniref100"
user="caobx"
filename="seq.fasta"
datadir="/public/home/caobx/benchmark/ExoNeg/pd_old"
outblast="seq.blast"

f_1=open(datadir+"/list.lst","r")
for lines in f_1.readlines():
	protein=lines.strip()
	name=protein
	outdir="/public/home/caobx/benchmark/ExoNeg/pd_old/"+name
	tag="PSSM_"+name
	
	f=open("mkpssm.py",'r')
	a=f.read()
	a=a.replace("!LIBDIR!",libdir)
	a=a.replace("!USER!",user)
	a=a.replace("!NAME!",name)
	a=a.replace("!TAG!",tag)
	a=a.replace("!OUTDIR!",outdir)
	a=a.replace("!DATADIR!",datadir)
	a=a.replace("!FILENAME!",filename)
	a=a.replace("!OUTBLAST!",outblast)
	f.close()
	
	logdir="/public/home/caobx/benchmark/ExoNeg/pd_old/"+name+"/log"
	if not os.path.exists(logdir):
		os.system("mkdir %s"%logdir)
	
	walltime="walltime=24:00:00"
	errfile=logdir+"/err_"+name
	outfile=logdir+"/out_"+name
	program=logdir+"/"+name+"_new.py"

	g=open(program,"w+")
	g.write(a)
	g.close()

	os.system("/public/home/caobx/bin/getQ.pl")
	os.system("chmod 777 %s" %program)
	os.system("qsub -e %s -o %s -l %s -N %s %s" %(errfile,outfile,walltime,tag,program))
	print("%s submitted\n" %program)
	time.sleep(2)