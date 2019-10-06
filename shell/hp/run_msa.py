#!/usr/bin/env python
# -*- coding: utf-8 -*-
import re
import os
import time

user = 'sry'
app = 'psiblast'
appdir = '/public/application/ncbi-blast-2.3.0+/bin/%s' %(app)
libdir = '/public/library/uniprot20/'
dataset_name = ['S1925', 'S2648']

for dataset in dataset_name:
	datadir = '/public/home/sry/projects/mCNN/datasets/%s/seq%s/' %(dataset, dataset)
	outblast = '/public/home/sry/mCNN/msa/psiblast/%s/' %(dataset)

	f_seq = open(datadir + '/list.lst', 'r') # pdbid_0001.fasta
	for lines in f_seq.readlines():
		filename = lines.strip()
		outdir = outblast + filename + '/'
		tag = app + '_' + filename

		f = open('msa_template.py', 'r')
		a = f.read()
		a = a.replace('!APPDIR!', appdir)
		a = a.replace('!LIBDIR!', libdir)
		a = a.replace('!DATADIR!', datadir)
		#a = a.replace('!NAME!', mut_id)
		a = a.replace('!USER!', user)
		a = a.replace('!TAG!', tag)
		a = a.replace('!OUTDIR!', outdir)

		a = a.replace('!FILENAME!', filename)
		a = a.replace('!OUTBLAST!', outblast)
		f.close()

		logdir = outdir + 'log/'
		if not os.path.exists(logdir):
			os.system('mkdir -p %s' % logdir)

		walltime = 'walltime = 24:00:00'
		errfile = logdir + 'err_' + filename
		outfile = logdir + 'out_' + filename
		run_psiblast = logdir  + filename + '.py'

		g = open(run_psiblast, 'w+')
		g.write(a)
		g.close()

		os.system('/public/home/sry/bin/getQ.pl')
		os.system('chmod 755 %s' % run_psiblast)
		os.system('qsub -e %s -o %s -l %s -N %s %s' % (errfile, outfile, walltime, tag, run_psiblast))
		print('%s submitted\n' % run_psiblast)
		time.sleep(1)
