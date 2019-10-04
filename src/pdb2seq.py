#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys, time
import warnings
from Bio import BiopythonWarning
from Bio.PDB.PDBParser import PDBParser
warnings.simplefilter('ignore', BiopythonWarning)

aa_dict = {'Ala': 'A', 'Arg': 'R', 'Asn': 'N', 'Asp': 'D', 'Cys': 'C',
           'Gln': 'Q', 'Glu': 'E', 'Gly': 'G', 'His': 'H', 'Ile': 'I',
           'Leu': 'L', 'Lys': 'K', 'Met': 'M', 'Phe': 'F', 'Pro': 'P',
           'Ser': 'S', 'Thr': 'T', 'Trp': 'W', 'Tyr': 'Y', 'Val': 'V'} # from wiki

def pdb2seq(seqname, filename, mdlid, chainid, wtflag, position='0', mtaa = '0'):
    lst = []
    parser = PDBParser(PERMISSIVE=1)
    structure = parser.get_structure(filename, './data/' + filename + '.pdb')
    model = structure[int(mdlid)]
    chain = model[chainid]
    if wtflag == 'WT':
        for residue in chain:
            res_id = residue.get_id()
            if res_id[0] == ' ':
                long_name = chain[res_id].get_resname()
                short_name = aa_dict[long_name[0]+long_name[1].lower()+long_name[2].lower()]
                lst.append(short_name)
    if wtflag=='MT':
        if position.isdigit():
            mutid = (' ',int(position),' ')
        else:
            mutid = (' ',int(position[:-1]),position[-1])
        for residue in chain:
            res_id = residue.get_id()
            if res_id[0] == ' ' and res_id != mutid:
                long_name = chain[res_id].get_resname()
                short_name = aa_dict[long_name[0]+long_name[1].lower()+long_name[2].lower()]
            elif res_id == mutid:
                short_name = mtaa
            lst.append(short_name)
    # print(lst)
    # print(len(set(lst)))
    fasta_name = '%s.fasta'%seqname
    g = open(fasta_name, 'w+')
    g.writelines('>%s.fasta|user:sry|date:%s|mdl:%s|chain:%s|pos:%s|mt_res:%s\n'
                 %(seqname, time.strftime("%a %b %d %H:%M:%S %Y",time.localtime()),
                   mdlid, chainid, position, mtaa))
    # print(lst)
    g.writelines(''.join(aa for aa in lst))
    g.close()

if __name__ == '__main__':
    # print(sys.argv[1:])
    seqname, filename, mdlid, chainid, wtflag, position, mtaa = sys.argv[1:]
    pdb2seq(seqname, filename, mdlid, chainid, wtflag, position, mtaa)