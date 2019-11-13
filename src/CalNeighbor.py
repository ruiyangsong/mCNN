#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
** Extract ATOM rows in pdb file, Calculate the distance from MT site, Deposit them to csv file **
** 10/10/2019.
** --sry.
'''
import os, argparse, warnings
import numpy as np
import pandas as pd
from Bio import BiopythonWarning
from Bio.PDB.PDBParser import PDBParser
from processing import str2bool

def CalNeighbor(PDBDIR, CHAIN, POSITION, MODEL = 0, HETATM = False, NUCLEIC  = False, center = 'CA', filter_res_atom=2):
    '''
    :param PDBDIR: PDB file dir.
    :param CHAIN: MT chain.
    :param POSITION: MT position.
    :param MODEL: Model number in pdb file.
    :param HETATM: If consider HETATM atoms.
    :param NUCIC: If consider NUCLEIC atoms.
    :center: Mutant site center ['CA','geometric'].
    :return: csv file which stored ATOM section and distances from MT_center.
    '''
    warnings.simplefilter('ignore', BiopythonWarning)

    if POSITION.isdigit():
        INODE = ' '
        POSID = int(POSITION)
    else:
        INODE = POSITION[-1]
        POSID = int(POSITION[:-1])
    MT_pos = (' ',POSID,INODE)

    aa_dict = {'Ala': 'A', 'Arg': 'R', 'Asn': 'N', 'Asp': 'D', 'Cys': 'C',
               'Gln': 'Q', 'Glu': 'E', 'Gly': 'G', 'His': 'H', 'Ile': 'I',
               'Leu': 'L', 'Lys': 'K', 'Met': 'M', 'Phe': 'F', 'Pro': 'P',
               'Ser': 'S', 'Thr': 'T', 'Trp': 'W', 'Tyr': 'Y', 'Val': 'V'}  # from wiki

    df_neighbor = pd.DataFrame({'chain': [], 'res': [], 'het': [], 'posid': [], 'inode': [], 'full_name': [],'atom_name':[],
                                'dist': [], 'x': [], 'y': [], 'z': [], 'occupancy': [], 'b_factor': []})
    pdbid = PDBDIR.split('/')[-1][:-4]
    parser = PDBParser(PERMISSIVE=1)
    structure = parser.get_structure(pdbid, PDBDIR)
    model = structure[MODEL]
    if center == 'CA':
        center_coord = model[CHAIN][MT_pos]['CA'].get_coord()
    elif center == 'geometric':
        atomcoordlst = [atom.get_coord() for atom in model[CHAIN][MT_pos] if not atom.get_name()[0] in ('H','D')]
        center_coord = np.array([0, 0, 0])
        for atomcoord in atomcoordlst:
            center_coord = center_coord + atomcoord
        center_coord = center_coord / len(atomcoordlst)
    for chain in model:
        chain_name = chain.get_id()
        res_id_lst = [res.get_id() for res in chain]
        print('The pdbid is:',pdbid)
        print('The chain is:',chain_name)
        print('The res_number in this chain is:', len(res_id_lst))
        if not HETATM:
            res_id_lst = list(filter(lambda tup: tup[0] == ' ', res_id_lst))  # NOT consider hetatm atoms.
        print('After filter NOT_HET, the total res_number is:',len(res_id_lst))
        if not NUCLEIC:
            res_id_lst = [tup for tup in res_id_lst if tup[0] == ' ' and chain[tup].get_resname() in [aa.upper() for aa in aa_dict.keys()]] # NOT consider nucleic acid atoms.
        print('After filter NOT_NUC, the total res_number is:',len(res_id_lst))
        # ## filter residue which contains only one atom.
        # res_id_lst = [tup for tup in res_id_lst if len(chain[tup]) > filter_res_atom]# res_id_lst = [tup for tup in res_id_lst if len(list(chain[tup].get_atoms())) > filter_res_atom]
        # print(MT_pos)
        # if chain_name == CHAIN:
        #     assert MT_pos in res_id_lst
        # print('After filter only one atom, the total res_number is:', len(res_id_lst))
        res_list = [chain[res_id] for res_id in res_id_lst]
        for res in res_list:
            res_name = res.get_resname()
            res_name = res_name[0]+res_name[1].lower()+res_name[2].lower()
            het, pos_id, inode = res.get_id()
            for atom in res:
                full_name, coord, occupancy, b_factor = atom.get_name(), atom.get_coord(), atom.get_occupancy(), atom.get_bfactor()
                name = full_name.strip()[0]
                if name == 'H' or name == 'D':
                    continue #H2O and D2O
                dist = np.linalg.norm(center_coord - coord)
                x,y,z = coord
                temp_array = np.array([chain_name,res_name,het,pos_id,inode,full_name,name,dist,x,y,z,occupancy,b_factor]).reshape(1, len(temp_array))
                temp_df = pd.DataFrame(temp_array)
                temp_df.columns = df_neighbor.columns
                df_neighbor = pd.concat([df_neighbor, temp_df], axis=0, ignore_index=True)
    print('The atom_number is:',len(df_neighbor))

    return df_neighbor, center_coord
def save_csv(DF, FILENAME, OUTDIR='.'):
    DF.to_csv('%s/%s.csv'%(OUTDIR,FILENAME), index=False)

if __name__ == '__main__':
    ## input parameters in shell.
    parser = argparse.ArgumentParser()
    parser.description = '* Calculate distances between Center at MT site with all the other atoms.\n' \
                         'Detailed introduction of pdb format refers to:\n' \
                         'http://www.cgl.ucsf.edu/chimera/docs/UsersGuide//tutorials/framepdbintro.html\n' \
                         'http://www.wwpdb.org/documentation/file-format-content/format33/v3.3.html'
    parser.add_argument('pdbdir',            type=str,  required=True,   help='The input directory of a .pdb file')
    parser.add_argument('chain',             type=str,  required=True,   help='The mutation chian')
    parser.add_argument('position',          type=str,  required=True,   help='The mutation position')
    parser.add_argument('filename',          type=str,  help='MT_pdbid_wtaa_chain_positon_mtaa')
    parser.add_argument('-o', '--outdir',    type=str,  default='.',     help='The output directory')
    parser.add_argument('-d', '--model',     type=str,  default='0',     help='Which model to extract atoms from, default=0')
    parser.add_argument('-het','--hetatm',   type=str,  default='False', help='Whether consider the HETATM atoms, default=False')
    parser.add_argument('-nuc', '--nucleic', type=str,  default='False', help='Whether consider the nucleic acid atoms, default=False')
    parser.add_argument('--filter',          type=int,  default=2,       help='The minimal number of atoms a residue have')
    parser.add_argument('-C','--center',     type=str,  default='CA',    choices=['CA','geometric'], help='MT center type, default = "CA"')
    args     = parser.parse_args()
    PDBDIR   = args.pdbdir
    CHAIN    = args.chain
    POSITION = args.position
    FILENAME = args.filename
    if args.outdir:
        OUTDIR = args.outdir
        if not os.path.exists(OUTDIR):
            os.mkdir(OUTDIR)
    if args.model:
        MODEL = int(args.model)
    if args.hetatm:
        HETATM = str2bool(args.hetatm)
    if args.nucleic:
        NUCLEIC = str2bool(args.nucleic)
    if args.filter:
        FILTER = args.filter
    if args.center:
        CENTER = args.center
    df, center_coord = CalNeighbor(PDBDIR, CHAIN, POSITION, MODEL, HETATM, NUCLEIC,center=CENTER, filter_res_atom=FILTER)
    save_csv(df, FILENAME, OUTDIR=OUTDIR)
    np.save('%s/center_coord.npy' % OUTDIR)