#!/usr/bin/env python
'''only standard residue in mt_chain are accepted'''
import os,warnings
import pandas as pd
from Bio.PDB import PDBIO, Select
from Bio import BiopythonWarning
from Bio.PDB.PDBParser import PDBParser

def main():
    mtcsvpth = '/public/home/sry/mCNN/dataset/TR/S2648_TR500.csv'
    df = pd.read_csv(mtcsvpth)# key, PDB, WILD_TYPE, CHAIN, POSITION, MUTANT, PH, TEMPERATURE, DDG
    df_pdb_chain = df.drop_duplicates(subset=['PDB','CHAIN'],keep='first')
    outdir = '/public/home/sry/mCNN/dataset/TR/pdb_chain'
    for i in range(len(df_pdb_chain)):
        pdbid,chain_name = df_pdb_chain.iloc[i,:][['PDB','CHAIN']].values
        pdbdir = '/public/home/sry/mCNN/dataset/TR/pdb/%s.pdb'%pdbid
        retrieve_pdb_chain(pdbdir, MDL=0, chain_name=chain_name, write=1, outpath=outdir)

def retrieve_pdb_chain(pdbdir,MDL=0,chain_name='A',write=0,outpath=None):
    warnings.simplefilter('ignore', BiopythonWarning)
    pdbid = pdbdir.split('/')[-1][0:4]
    parser = PDBParser(PERMISSIVE=1)
    structure = parser.get_structure(pdbid, pdbdir)
    model = structure[MDL]
    if write == 1:
        if outpath == None:
            raise RuntimeError('out path is None!')
        os.makedirs(outpath,exist_ok=True)
        class ModelSelect(Select):
            def accept_model(self, model):
                if model.get_id() == 0:
                    return True
                else:
                    return False

            def accept_chain(self, chain):
                """Overload this to reject chains for output."""
                if chain.get_id() == chain_name:
                    return True
                else:
                    return False

            def accept_residue(self, residue):
                if residue.get_id()[0] == ' ':
                    return True
                else:
                    return False

            def accept_atom(self, atom):
                """Overload this to reject atoms for output."""
                return 1
        io = PDBIO()
        io.set_structure(structure)
        io.save('%s/%s.pdb' % (outpath,pdbid), ModelSelect(), preserve_atom_numbering=True)
    return model

if __name__ == '__main__':
    main()