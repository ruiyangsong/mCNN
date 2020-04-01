#!/usr/bin/env /usr/bin/python


from pyrosetta import *
import re,sys
import os, shutil
import random
import numpy as np
import pickle
import math
os.environ["OPENBLAS_NUM_THREADS"] = "1"

phi=[]
psi=[]
phi_prob=[]
psi_prob=[]

#exit()
pcut = float(sys.argv[1])
k=int(sys.argv[2])

if(os.path.isfile("phipsi.npz")):
    npz=np.load("phipsi.npz")
    phi=npz['phi']
    phi_prob=npz['phi_prob']
    psi=npz['psi']
    psi_prob=npz['psi_prob']

def main():
    #print pcut, k
    
    init('-mute all -hb_cen_soft -relax:default_repeats 5 -default_max_cycles 200')

    #read fasta sequence
    sequence=read_fasta("seq.fasta")
    nres=len(sequence)
    
    scorefxn=ScoreFunction()
    scorefxn.set_weight(rosetta.core.scoring.cen_hb, 5.0)    # short-range hbonding
    scorefxn.set_weight(rosetta.core.scoring.rama, 1.0)    # ramachandran score
    scorefxn.set_weight(rosetta.core.scoring.omega, 0.5)    # omega torsion score
    scorefxn.set_weight(rosetta.core.scoring.vdw, 1.0)
    scorefxn.set_weight(rosetta.core.scoring.atom_pair_constraint, 5)    
    scorefxn.set_weight(rosetta.core.scoring.dihedral_constraint, 4)
    scorefxn.set_weight(rosetta.core.scoring.angle_constraint, 4)

    
    scorefxn1=ScoreFunction()
    scorefxn1.set_weight(rosetta.core.scoring.cen_hb, 5.0)    # short-range hbonding
    scorefxn1.set_weight(rosetta.core.scoring.rama, 1.0)    # ramachandran score
    scorefxn1.set_weight(rosetta.core.scoring.omega, 0.5)    # omega torsion score
    scorefxn1.set_weight(rosetta.core.scoring.vdw, 3.0)
    
    scorefxn1.set_weight(rosetta.core.scoring.atom_pair_constraint, 3)
    scorefxn1.set_weight(rosetta.core.scoring.dihedral_constraint, 1)
    scorefxn1.set_weight(rosetta.core.scoring.angle_constraint, 1)

  
    scorefxn_vdw=ScoreFunction()
    scorefxn_vdw.set_weight(rosetta.core.scoring.vdw, 1.0)
    scorefxn_vdw.set_weight(rosetta.core.scoring.rama, 1.0)

    scorefxn_cart=ScoreFunction()
    #scorefxn_cart.set_weight(rosetta.core.scoring.cen_hb, 5.0)    # short-range hbonding
    scorefxn_cart.set_weight(rosetta.core.scoring.hbond_sr_bb, 3.0)    # short-range hbonding
    scorefxn_cart.set_weight(rosetta.core.scoring.hbond_lr_bb, 3.0)    # long-range hbonding
    scorefxn_cart.set_weight(rosetta.core.scoring.rama, 1.0)    # ramachandran score
    scorefxn_cart.set_weight(rosetta.core.scoring.omega, 0.5)    # omega torsion score
    scorefxn_cart.set_weight(rosetta.core.scoring.vdw, 0.5)
    scorefxn_cart.set_weight(rosetta.core.scoring.cart_bonded, 0.1)
    
    scorefxn_cart.set_weight(rosetta.core.scoring.atom_pair_constraint, 5)   
    scorefxn_cart.set_weight(rosetta.core.scoring.dihedral_constraint, 4)
    scorefxn_cart.set_weight(rosetta.core.scoring.angle_constraint, 4) 


    mmap = MoveMap()
    mmap.set_bb(True)
    mmap.set_chi(False)
    mmap.set_jump(True)



    n_iter = 1000 + nres
    print(n_iter)

    min_mover = rosetta.protocols.minimization_packing.MinMover(mmap, scorefxn, 'lbfgs_armijo_nonmonotone', 0.0001, True)
    min_mover.max_iter(n_iter)

    min_mover1 = rosetta.protocols.minimization_packing.MinMover(mmap, scorefxn1, 'lbfgs_armijo_nonmonotone', 0.0001, True)
    min_mover1.max_iter(n_iter)

    min_mover_vdw = rosetta.protocols.minimization_packing.MinMover(mmap, scorefxn_vdw, 'lbfgs_armijo_nonmonotone', 0.0001, True)
    min_mover_vdw.max_iter(500)
    
    min_mover_cart=rosetta.protocols.minimization_packing.MinMover(mmap, scorefxn_cart, 'lbfgs_armijo_nonmonotone', 0.0001, True)
    min_mover_cart.max_iter(n_iter)
    min_mover_cart.cartesian(True)

    
    repeat_mover = RepeatMover(min_mover, 3)

   
    count=0
    score_file="cen_"+str(pcut)+"_"+str(k)+".sc"
    score_file_r="score_r.sc"


#    ref(score_file, score_file_r, nres)
#    select_model(score_file_r)    
#    exit()


    
    SS = open(score_file, "w")
    SS.close() 

    rep_all = [] 
    if(os.path.isfile("repul_cst")):
        rep_all=read_cst("repul_cst")
    
    cst_all = read_cst("cst.txt")

    

    topcut=0.65
    cst_for_score=[]
    while(len(cst_for_score)<nres and topcut>0):
        topcut -= 0.05
        cst_for_score=fetch_cst_high(cst_all, nres, 1, 10000, topcut, 1)


    sep1=1
    sep2=10000
    cst_top=[]
    pcut1=0.85 #used for generating starting conformtaion
    while(len(cst_top)<nres and pcut1>0.1):
        pcut1 -= 0.05
        cst_top=fetch_cst_high(cst_all, nres, sep1, sep2, pcut1, 1)

        
    if(len(cst_for_score)<nres):
        print("warning: not enough reliable constraints to use: ", len(cst_for_score))

    print("num of top cst:", len(cst_for_score))
    print("k, pcut:", k, pcut)
    pose0=make_start(k, sequence, scorefxn_vdw, min_mover_vdw)

    #0.0, equals to the paper version
    
    probCuts=[0.0, 0.05, 0.1, 0.2] #probability cutoff for selecting the low-probability restraints, useful for avoiding clash for big proteins
    totcut=len(probCuts)
    for it in range(0, 4):
        low_cut=probCuts[it]
        if(pcut<low_cut): 
            print("warning: pcut<low_cut, ", pcut, low_cut)
            if(it>0): 
                print("skip iteration with low_cut", it)
                continue
            
        print("low_cut=", low_cut)
        
        pose=Pose()
        pose.assign(pose0)

        sep1=1
        sep2=10000
        cst_rep_low = fetch_cst_low(cst_all, sep1, sep2, 0.0, low_cut)
        print("repulsive cst: ", len(cst_rep_low))
   
        add_cst(pose, cst_rep_low)


        print("\nminimize with short cst...\n")
        #short
        sep1=1
        sep2=12
        
        cst_short=fetch_cst(cst_all, nres, sep1, sep2, pcut, 1)
        run_min(cst_short, 1, pose, repeat_mover, min_mover_cart)
        clash_score=remove_clash(scorefxn_vdw, min_mover1, pose)
        pose.dump_pdb("s.pdb")
        
        #medm
        print("\nminimize with medm cst...\n")
        sep1=12
        sep2=24
        cst_medm=fetch_cst(cst_all, nres, sep1, sep2, pcut, 1)
        run_min(cst_medm, 1, pose, repeat_mover, min_mover_cart)
        clash_score=remove_clash(scorefxn_vdw, min_mover1, pose)       
        pose.dump_pdb("m.pdb")
        
        #long
        
        print("\nminimize with long cst...\n")
        sep1=24
        sep2=10000
        
        #add_cst(pose, cst_rep_low)

        cst_long=fetch_cst(cst_all, nres, sep1, sep2, pcut, 1)
        run_min(cst_long, 1, pose, repeat_mover, min_mover_cart)
        clash_score=remove_clash(scorefxn_vdw, min_mover1, pose)           
    
        
        name="pose" +  str(k) + "_" + str(pcut) + "_" + str(it) + ".pdb"
        output_data(pose, cst_for_score, scorefxn, name, score_file)

    
        #generate alternative models for the first start
        print("generate alternative model with short+medium, then long")
        if(k==0):
            pose.assign(pose0)
        else:
            pose=make_start(k, sequence, scorefxn_vdw, min_mover_vdw) #use new conformation to increase diversity




        #print "prepare start conformation...\n"
        sep1=1
        sep2=10000

        print("prepare start conformation with top restraints, and repulsive restraints")

        pose0_with_topcst=Pose()
        pose0_with_topcst.assign(pose0)
        add_cst(pose0_with_topcst, cst_rep_low)
        run_min(cst_top, 1, pose0_with_topcst, repeat_mover, min_mover_cart)
        remove_clash(scorefxn_vdw, min_mover1, pose0_with_topcst)
        pose0_with_topcst.dump_pdb("ini.pdb")
        pose0_with_topcst.remove_constraints() #remmove constraints to avoid duplicated ones

        
        pose.assign(pose0_with_topcst) 
        add_cst(pose, cst_rep_low)

        #short+medm first, and then replusive+long
        sep1=1
        sep2=24
        cst=fetch_cst(cst_all, nres, sep1, sep2, pcut, 1)
        run_min(cst, 1, pose, repeat_mover, min_mover_cart)
        remove_clash(scorefxn_vdw, min_mover1, pose)
        pose.dump_pdb("ini_m.pdb") 
    
    
        #long
        sep1=24
        sep2=10000
        cst=fetch_cst(cst_all, nres, sep1, sep2, pcut, 1)
        run_min(cst, 1, pose, repeat_mover, min_mover_cart)
        clash_score=remove_clash(scorefxn_vdw, min_mover1, pose)


        #if(clash_score>100):
        #    clash_score=remove_clash_with_repusive_cst(scorefxn_vdw, cst_rep, cst_rep_low, min_mover1, repeat_mover, min_mover_cart, pose)

    
        name="pose" +  str(k) + "_" + str(pcut) + "_0_" + str(it)+ ".pdb"
        output_data(pose, cst_for_score, scorefxn, name, score_file)
    
    
        #using all cst
        print("generate alternative model1 with all cst together")

        pose.assign(pose0_with_topcst)
        sep1=1
        sep2=10000
        cst=fetch_cst(cst_all, nres, sep1, sep2, pcut, 1)
        add_cst(pose, cst_rep_low)
        #add_cst(pose, rep_all)
        run_min(cst, 1, pose, repeat_mover, min_mover_cart)
        clash_score=remove_clash(scorefxn_vdw, min_mover1, pose)

        #if(clash_score>100):
        #    clash_score=remove_clash_with_repusive_cst(scorefxn_vdw, cst_rep, cst_rep_low, min_mover1, repeat_mover, min_mover_cart, pose)
    
        name="pose" +  str(k) + "_" + str(pcut) + "_1_" + str(it) + ".pdb"
        output_data(pose, cst_for_score, scorefxn, name, score_file)            
    #it
  #  ref(score_file, score_file_r, nres)
  #  select_model(score_file_r)    
#end of main



def ref(filename, filename_r, nres):    
    probCuts=[0.15, 0.1, 0.05] #probability cutoff for selecting the restraints

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

    topmodels=[]
    for prob in probCuts:
        selected=fetch_subset(decoy, sc, prob)
        topmodels += selected

       
    ##start refinement

    #print "here";
    SS = open(filename_r, "w")
    SS.close()
    scorefxn_fa=create_score_function('ref2015')
    scorefxn_fa.set_weight(rosetta.core.scoring.atom_pair_constraint, 3)

    mmap = MoveMap()
    mmap.set_bb(True)
    mmap.set_chi(True)
    mmap.set_jump(True)

    relax=rosetta.protocols.relax.FastRelax()
    relax.set_scorefxn(scorefxn_fa)
    relax.max_iter(200)
    relax.dualspace(True)
    relax.set_movemap(mmap)

    cstname="cst_good.txt"
    if(os.path.isfile(cstname)):pass
    else:cstname="cst.txt"

    cst_all = read_cst(cstname)

    sep1=3
    sep2=10000
    pcut=0.15
    cst=fetch_cst(cst_all, nres, sep1, sep2, pcut, 1)


    for model in topmodels:    

        pose = pose_from_file(model + ".pdb")
        add_cst(pose, cst)

        print("refine model", model)
        relax.apply(pose)

        SS = open(filename_r, "a")
        name=model + "_ref.pdb";
        inf=name + "\t" + str(scorefxn_fa(pose))
        SS.write(inf)
        SS.write("\n")
        pose.dump_pdb(name)
        SS.close()
    

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
    

def output_data(pose, cst, scorefxn, name, filename):
    pose.remove_constraints()
    add_cst(pose, cst)
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
        for nm in range(0, 2):
            mover.apply(pose)
            clash_score=float(scorefxn(pose))
            print("clash_score=", clash_score)
            if(clash_score<10): break
            
    return clash_score

def remove_clash_with_repusive_cst(scorefxn, cst1, cst2, mover_vdw, mover_min, mover_cart, pose):
 
    print("heavy clash, add regular repulsive restraints and minimize structure")
    add_cst(pose, cst1)
    mover_min.apply(pose)
    mover_cart.apply(pose)
    clash_score=remove_clash(scorefxn, mover_vdw, pose)

    if(clash_score>100):
        print("heavy clash still there, add strong repulsive restraints and minimize structure", clash_score)
        add_cst(pose, cst2)
        mover_min.apply(pose)
        mover_cart.apply(pose)
        clash_score=remove_clash(scorefxn, mover_vdw, pose)

        if(clash_score>100):
            print("heavy clash still there, please check the folding by hand", clash_score)
            
    return clash_score

    




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
        if(b[0]=="Angle"): pcut=cut+0.5
        if(b[0]=="AtomPair"): pcut=cut

        if(pcut>0.9): pcut=0.9
 
        sep=abs(j-i)
        if(sep<sep1 or sep >=sep2): continue
        #print(dcst[3])
        if(flag==1):
            if(float(dcst[3])>=pcut):
                array.append(line)
                #print line
        else:
            if(float(dcst[3])<pcut):
                array.append(line)
            
    return array



def fetch_cst_high(cst, nres, sep1, sep2, cut, flag):
    pcut=cut
    array=[]
    for line in cst:
        #print line
        line=line.rstrip()
        b=re.split("\s+", line)

        if(b[0]!="AtomPair"): continue

        #print b[1],b[2],b[3],b[4]
        m=re.search('(?<=#).+', line)
        dcst=re.split("\s+", m.group(0))
        i=int(b[2])
        j=int(b[4])

        if(j==i):j=int(b[6]) #omega, phi
        if(j==i):j=int(b[8]) #theta

        sep=abs(j-i)
        if(sep<sep1 or sep >=sep2): continue
        #print(dcst[3])
        if(flag==1):
            if(float(dcst[3])>=pcut):
                array.append(line)
                #print line
        else:
            if(float(dcst[3])<pcut):
                array.append(line)

    return array



def fetch_cst_low(cst, sep1, sep2, lb, ub):

    array=[]
    for line in cst:
        #print line
        line=line.rstrip()
        b=re.split("\s+", line)
        #print b[1],b[2],b[3],b[4]

        if(b[0]!="AtomPair"): continue        
        m=re.search('(?<=#).+', line)
        dcst=re.split("\s+", m.group(0))
        i=int(b[2])
        j=int(b[4])

        if(j==i):j=int(b[6]) #omega, phi
        if(j==i):j=int(b[8]) #theta


        sep=abs(j-i)
        if(sep<sep1 or sep >=sep2): continue
        #print(dcst[3])
        if(float(dcst[3])>=lb and float(dcst[3])<ub):
            array.append(line)
                #print line

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

    b=np.shape(phi)
    pose=pose_from_sequence(sequence, 'centroid' )
    if(k<=1 and b[0]>0):            
        set_predicted_dihedral(pose, k)
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

def set_predicted_dihedral(pose, k):
    b=np.shape(phi)
    #print phi
    #print b[0]
    for i in range(0, b[0]):
        prob=phi_prob[i]
        dih=phi[i]
        m=select_phipsi(prob, k)
        #print m
        v=float(dih[m])
        pose.set_phi(i+1, v)

        prob=psi_prob[i]
        dih=psi[i]
        m=select_phipsi(prob, k)
        v=float(dih[m])
        pose.set_psi(i+1, v)

    #exit()
    return(pose)


def select_phipsi_(prob, k):
    n=len(prob)
    m=0
    if(k==0):return 0
    else:
        r=random.random()
        ub=0
        lb=ub
        for i in range(0, n):
            ub += prob[i]                        
            if(r>=lb and r<ub):
                m=i
                break

    return m

def select_phipsi(prob, k):
    n=len(prob)
    m=0
    if(k==0):return 0
    else:
        r=random.random()
        if(r>0.8): m=1

    return m

    
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
    tmpname=str(pcut)+"_"+str(k)+"_"+"tmp.sp";
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
