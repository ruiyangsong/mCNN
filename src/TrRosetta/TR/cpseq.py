#!/usr/bin/env python
import os
def cpseq_from_list(listpth,msadir,outdir):
    wild_dir = msadir+'/mdl0_wild'
    mutant_dir = msadir+'/mdl0_mutant'
    with open(listpth) as f:
        lines = f.readlines()
        for line in lines:
            try:
                key,PDB,CHAIN = line.strip().split('.')
                wild_tag = PDB+'/'+CHAIN
                seqpth = '%s/%s/%s_%s.fasta'%(wild_dir,wild_tag,PDB,CHAIN)
                wild_tag_outdir = '%s/%s'%(outdir,line.strip())
                if not os.path.exists(wild_tag_outdir):
                    os.makedirs(wild_tag_outdir)
                os.system('cp %s %s/seq.fasta'%(seqpth,wild_tag_outdir))
            except:
                key,PDB,WILD_TYPE,CHAIN,POSITION,MUTANT = line.strip().split('.')
                mutant_tag = PDB+'_'+WILD_TYPE+'_'+CHAIN+'_'+POSITION+'_'+MUTANT
                seqpth = '%s/%s/%s_%s.fasta' % (mutant_dir, mutant_tag, PDB, CHAIN)
                mutant_tag_outdir = '%s/%s' % (outdir,line.strip())
                if not os.path.exists(mutant_tag_outdir):
                    os.makedirs(mutant_tag_outdir)
                os.system('cp %s %s/seq.fasta' % (seqpth, mutant_tag_outdir))

if __name__ == '__main__':
    # listpth = 'E:/projects/mCNN/ieee_access/supplymentary_data/datasets/TR/list'
    listpth = '/public/home/sry/mCNN/dataset/TR/list'
    msadir = '/public/home/sry/mCNN/dataset/S2648/feature/msa'
    outdir = '/public/home/sry/mCNN/dataset/TR/output'
    cpseq_from_list(listpth,msadir,outdir)