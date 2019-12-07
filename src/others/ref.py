#!/usr/local/bin/python

from pyrosetta import *
import re,sys
import os, shutil
import random
import numpy as np
import pickle
import math
os.environ["OPENBLAS_NUM_THREADS"] = "1"


#exit()
#datadir = sys.argv[1]
initial_model = sys.argv[1]
mut_id=int(sys.argv[2])
mut_res=sys.argv[3]


def main():
    
    init('-hb_cen_soft -relax:default_repeats 1 -default_max_cycles 50')

#    os.chdir(datadir)
    
    #print "here";

    scorefxn_fa=create_score_function('ref2015')
    scorefxn_fa.set_weight(rosetta.core.scoring.atom_pair_constraint, 5)
    scorefxn_fa.set_weight(rosetta.core.scoring.dihedral_constraint, 1)
    scorefxn_fa.set_weight(rosetta.core.scoring.angle_constraint, 1)


    mmap = MoveMap()
    mmap.set_bb(True)
    mmap.set_chi(True)
    mmap.set_jump(True)

    relax=rosetta.protocols.relax.FastRelax()
    relax.set_scorefxn(scorefxn_fa)
    relax.max_iter(200)
    relax.dualspace(True)
    relax.set_movemap(mmap)

    
    #refine without cst to estimate model accuracy
    pose = pose_from_file(initial_model + ".pdb")

    print("refine without cst")
    relax.apply(pose)
    sfile="ref0_" + initial_model+"_.sc"
    SS = open(sfile, "w")
    name=initial_model  + "_ref.pdb";
    inf=name + "\t" + str(scorefxn_fa(pose))
    SS.write(inf)
    SS.write("\n")
    pose.dump_pdb(name)

    mutator = rosetta.protocols.simple_moves.MutateResidue(mut_id,mut_res)
    mutator.apply(pose)
    #pose.dump_pdb("mut.pdb")

    #refine mutated structure
    relax.apply(pose)
    name=initial_model  + "_mut.pdb";
    inf=name + "\t" + str(scorefxn_fa(pose))
    SS.write(inf)
    SS.write("\n")
    pose.dump_pdb(name)

    SS.close()


#end of main


def select_model(filename):
    sc=[]
    decoy=[]
    i=0
    with open(filename, "r") as f:
        for line in f:
            line=line.rstrip()
            if(line[0] == "#"):
                continue
            else:
                b=re.split("\t", line)
                
                sc.append(float(b[1]))
                decoy.append(b[0])
                #print b[0], b[1]

    idx=np.argsort(sc)
    topN=5
    if(topN>len(idx)): topN=len(idx)
    final=[]
    k=1
    for i in idx:
        #print decoy[i]
        if(not os.path.isfile(decoy[i])): continue
        name="model"+str(k)+".pdb"
        shutil.copy(decoy[i], name)
        k +=1
        if(k>5): break 


                
                
def fetch_subset(decoys, score, prob):
    nd=len(decoys)
    selected_decoys=[]
    selected_scores=[]
    for i in range(0, nd):
        d=decoys[i]
        Reg=re.compile(r'(.+).pdb')
        mo=Reg.search(d)
        #print mo.group(1)
        rst=re.split("_", mo.group(1))
        if(float(rst[1])==prob):
            selected_decoys.append(mo.group(1))
            selected_scores.append(score[i])

            
    idx=np.argsort(selected_scores)
    topN=10
    if(topN>len(idx)): topN=len(idx)
    final=[]
    for j in range(0,topN):
        i=idx[j]
        final.append(selected_decoys[i])
        #print selected_decoys[i], selected_scores[i]
    return final;
    

def output_data(pose, scorefxn, name, filename):
    inf=name + "\t" + str(scorefxn(pose))
    SS = open(filename, "a")
    SS.write(inf)
    SS.write("\n")
    SS.close()
    
    pose.dump_pdb(name)


def remove_clash(scorefxn, mover, pose):
    clash_score=float(scorefxn(pose))
    print("clash_score=", clash_score)
    if(clash_score>10):
        for nm in range(0, 5):
            mover.apply(pose)
            clash_score=float(scorefxn(pose))
            print("clash_score=", clash_score)
            if(clash_score<10): break
            


def run_min(cst_all, n_sets, pose, mover1, mover2):
    if(len(cst_all)==0):
        print("warning: empty constraint set")
        return
    
    random.shuffle(cst_all)    
    b_size=int(len(cst_all)/n_sets)                                                                                                                    
    for i in range(0, len(cst_all), b_size):
        batch=cst_all[i:i+b_size]
        add_cst(pose, batch)
        mover1.apply(pose)
        mover2.apply(pose)


def fetch_cst(cst, nres, sep1, sep2, cut, flag):
    pcut=cut
    array=[]
    for line in cst:
        #print line
        line=line.rstrip()
        b=re.split("\s+", line)
        #print b[1],b[2],b[3],b[4]             
        m=re.search('(?<=#).+', line)
        dcst=re.split("\s+", m.group(0))
        i=int(b[2])
        j=int(b[4])
        
        if(j==i):j=int(b[6]) #omega, phi
        if(j==i):j=int(b[8]) #theta
        if(b[0]=="Dihedral"): pcut=cut+0.5
        if(b[0]=="Angle"): pcut=cut+0.6
        if(b[0]=="AtomPair"): pcut=cut
    
    
        sep=abs(j-i)
        if(sep<sep1 or sep >=sep2): continue
        #print dcst[3]
        if(flag==1):
            if(float(dcst[3])>=pcut):
                array.append(line)
                #print line
        else:
            if(float(dcst[3])<pcut):
                array.append(line)
            
    return array

def compute_dist(pose, i, atmi, j, atmj):

    aai=pose.residue(i)
    xyz_i=aai.xyz(atmi)
    
    aaj=pose.residue(j)
    xyz_j=aaj.xyz(atmj)
    d=dist(xyz_i, xyz_j)
    return d

def get_pairwise_dist(pose):
    nres=pose.total_residue()
    for i in range(1, nres-2):
        aai=pose.residue(i)
        resi=aai.name()
        resi=resi[0:3]
        atmi="CB"
        if(resi == "GLY"):atmi="CA"
        xyz_i=aai.xyz(atmi)
        for j in range(i+3, nres+1):
            aaj=pose.residue(j)
            resj=pose.residue(j).name()
            resj=resj[0:3]
            atmj="CB"
            if(resj == "GLY"):atmj="CA"
            xyz_j=aaj.xyz(atmj)
            d=dist(xyz_i, xyz_j)
            if(d<20):
                pass
                #print i, j, d
        #end j
    #end i
                    
def dist(x, y):
    cut=20
    d=100
    a=abs(x[0]-y[0]);
    b=abs(x[1]-y[1]);
    c=abs(x[2]-y[2]);
    if(a>cut or b>cut or c>cut):
        return d
    else:
        d=math.sqrt(a**2+b**2+c**2)
        return d
        
    
def apply_cst(cst_file, b_size, pose, mover1, mover2):
    cst_all=read_cst(cst_file)
    if(b_size==0):
        b_size=len(cst_all)
    else:
        random.shuffle(cst_all)
                                                                                                                    
    for i in range(0, len(cst_all), b_size):
        batch=cst_all[i:i+b_size]          
        add_cst(pose, batch)
        #print(pose.constraint_set())
        mover1.apply(pose)
        mover2.apply(pose)        
        #print(mover.num_accepts(), mover.acceptance_rate())

    return cst_all


def make_start(k, sequence, scorefxn, mover):

    pose=pose_from_sequence(sequence, 'centroid' )
    if(os.path.isfile("tor_ss.dat") and k<1):               
        set_predicted_dihedral("tor_ss.dat", pose)
    else:
        set_random_dihedral(pose)

    clash_score=float(scorefxn(pose))
    if(clash_score>10):
        for nm in range(0, 5):
            mover.apply(pose)
            clash_score=float(scorefxn(pose))
            print("clash_score=", clash_score)
            if(clash_score<10): break
           
    return pose

def set_predicted_dihedral(filename,pose):
    with open(filename, "r") as f:
        for line in f:
            if(line[0] == "#"):
                continue
            else:
                b=re.split("\t", line)                
                pose.set_phi(int(b[0]),float(b[2]))
                pose.set_psi(int(b[0]),float(b[3]))
                if(b[1]=="C"):b[1]="L"
                pose.set_secstruct(int(b[0]), b[1])
    return(pose)

def convert_spx(filename):
    out=open("tor_ss.dat", "w")
    out.write("#ID\tSS\tPHI\tPSI\n")
    with open(filename, "r") as f:
        for line in f:
            if(line[0] == "#"):
                continue
            else:
                b=re.split("\s+", line)
               
                out.write("%d\t%c\t%.1f\t%.1f\n" %(int(b[1]), b[3], float(b[4]), float(b[5])))

    out.close()

def set_random_dihedral(pose):
    nres = pose.total_residue()
    for i in range(1, nres):
	#pick phi/psi randomly from:
        #-140  153 180 0.135 B
	# -72  145 180 0.155 B
        #-122  117 180 0.073 B
	# -82  -14 180 0.122 A
	# -61  -41 180 0.497 A
	#  57   39 180 0.018 L
        phi,psi=random_dihedral()        
        pose.set_phi(i,phi)
        pose.set_psi(i,psi)
        pose.set_omega(i,180)

    return(pose)

def random_dihedral():
    phi=0
    psi=0
    r=random.random()
    if(r<=0.135):
        phi=-140
        psi=153
    elif(r>0.135 and r<=0.29):
        phi=-72
        psi=145
    elif(r>0.29 and r<=0.363):
        phi=-122
        psi=117
    elif(r>0.363 and r<=0.485):
        phi=-82
        psi=-14
    elif(r>0.485 and r<=0.982):
        phi=-61
        psi=-41
    else:
        phi=57
        psi=39
    return(phi, psi)
        
def read_cst(file):
    array=[]
    with open(file, "r") as f:
        for line in f:
            #print line
            line=line.rstrip()
            array.append(line)
    return array


def add_cst(pose, array):
    constraints = rosetta.protocols.constraint_movers.ConstraintSetMover()
    tmpname=initial_model+"_"+"tmp.sp";
    F = open(tmpname, "w")
    for a in array:
        F.write(a)
        F.write("\n")
    F.close()
    constraints.constraint_file(tmpname)
    constraints.add_constraints(True)
    constraints.apply(pose)
    os.remove(tmpname)

        
def read_fasta(file):
    fasta="";
    with open(file, "r") as f:
        for line in f:
            if(line[0] == ">"):
                continue
            else:
                line=line.rstrip()
                fasta = fasta + line;
    return fasta
  

if __name__ == '__main__':  
    main()
