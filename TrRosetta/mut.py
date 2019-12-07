import sys,os,json
import tempfile
import numpy as np

from arguments import *
from utils_ros import *
from pyrosetta import *
from pyrosetta.rosetta.protocols.minimization_packing import MinMover

os.environ["OPENBLAS_NUM_THREADS"] = "1"

def main():

    init('-hb_cen_soft -relax:default_repeats 5 -default_max_cycles 200 -out:level 100')

    pose = pose_from_file("template.pdb")


    mutator = rosetta.protocols.simple_moves.MutateResidue(21,'TRP')
    mutator.apply(pose)
    pose.dump_pdb("mut.pdb")

if __name__ == '__main__':
    main()
