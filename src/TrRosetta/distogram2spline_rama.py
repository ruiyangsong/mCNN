#!/usr/bin/env /usr/bin/python

import sys
import os
import pickle
from math import exp,log,sqrt,pi,pow
import numpy as np
#Parameters
PCUT = 0.001 #Sum of probability at 4~20 Angstrom to decide whether to use the residue pair,
PCUT1= 0.5
EBASE = -0.05
EREP = [10.0,3.0,0.5] #Repulsion penalty at 0.0, 2.0, 3.0 Angstrom
PREP = 0.1 #Sum of probabilty at 4~20 to Angstrom decide whether should be apart (i.e. apply sigmoid-repulsion)
SIGD,SIGM = (10.0, 1.0)
bin_p=0.0
bin_p1=0.01 #for distance and std estimation
MEFF=0.0001
DCUT=19.75
ALPHA=1.57 #1.61
pid=0.01744 #3.14/180
power=0.2

def main():
    npzfile =sys.argv[1]
    ramafile =sys.argv[2]
    refpdb = sys.argv[3]
#    stage = sys.argv[4]

    
    npz=np.load(npzfile)

    ress = read_aas(refpdb)
   
    if not os.path.exists("splines"): os.mkdir("splines")

    cstout = open("cst.txt", "w")
    
    dist_cst(npz,ress,cstout, "CB")    

    omega_cst(npz,ress,cstout)
    theta_cst(npz,ress,cstout)
    phi_cst(npz,ress,cstout)

    npz_field=npz.files
    if('dist2' in npz_field):
        dist_cst(npz,ress,cstout, "CA")

    if(os.path.isfile(ramafile)): 
        rama=np.load(ramafile)
        phi, phi_prob=rama_phi_cst(rama,ress,cstout)
        psi, psi_prob=rama_psi_cst(rama,ress,cstout)

        np.savez("phipsi.npz", phi=phi, phi_prob=phi_prob, psi=psi, psi_prob=psi_prob)

    #exit()
    cstout.close()
            
#end main


def rama_phi_cst(npz, ress, cstout):
    phi=npz['phi']

    
    #cstout1 = file("phipsi.txt", "w")

    nres=len(ress)
    top_phi=np.zeros((nres, 3))
    top_prob=np.zeros((nres, 3))

    for i,aa1 in enumerate(ress):

        atm1 = 'C'
        atm2 = 'N'
        atm3 = 'CA'
        atm4 = 'C'
  
                        
        #check probability
        Praw = phi[0][i]    
            
        first_bin=0
        first_d=-175
            
        Pnorm = Praw

        idx=np.argsort(-Praw)
        for t in range(0, 3):
            k=idx[t]
            d = first_d + k*10
            top_phi[i][t]=d
            top_prob[i][t]=Praw[k]
            #print k, d, Praw[k]
        #exit();

        #print "\n"


        xs = []
        ys = []
        weight=0.0
        for P in Pnorm:
            weight += P
            
        #if weight<0.2: continue

        Pref = Pnorm[-1]+MEFF
        for k,P in enumerate(Pnorm):                            
            d = pid*(first_d + k*10)
                
            #dnorm=Pnorm_bkgr[k]+MEFF
            dnorm=Pref
            E=0
            E = -log((P+MEFF)/dnorm)

            xs.append(d)
            ys.append(E)                
        #end for k
            
        #name="splines/%d_psi.txt"%(i+1)
        #out = open(name,'w')
        #out.write('x_axis'+'\t%7.2f'*len(xs)%tuple(xs)+'\n')
        #out.write('y_axis'+'\t%7.3f'*len(ys)%tuple(ys)+'\n')
        #out.close()

        #format: AtomPair CB 13 CB 37 SPLINE tag fname expt? weight binsize
        #form = 'Dihedral %s %d %s %d %s %d %s %d SPLINE TAG %s 1.0 1.0 0.1744 #0.0 0.0 0.0 %.3f\n'
        #j=i+1
        #if(i>0):cstout.write(form%(atm1, j-1, atm2, j, atm3, j, atm4, j, name, weight))

    #end i
    #cstout.close()
    return top_phi, top_prob
#end rama_phi


def rama_psi_cst(npz, ress, cstout):
    phi=npz['psi']
    
    #cstout = file("cst_theta.txt", "w")

    nres=len(ress)
    top_phi=np.zeros((nres, 3))
    top_prob=np.zeros((nres, 3))

    for i,aa1 in enumerate(ress):

        atm1 = 'N'
        atm2 = 'CA'
        atm3 = 'C'
        atm4 = 'N'
  
                        
        #check probability
        Praw = phi[0][i]    
            
        first_bin=0
        first_d=-175
            
        Pnorm = Praw
        idx=np.argsort(-Praw)

        for t in range(0, 3):
            k=idx[t]
            d = first_d + k*10
            top_phi[i][t]=d
            top_prob[i][t]=Praw[k]


        xs = []
        ys = []
        weight=0.0
        for P in Pnorm:
            weight += P
            
        #if weight<0.2: continue

        Pref = Pnorm[-1]+MEFF
        for k,P in enumerate(Pnorm):                            
            d = pid*(first_d + k*10)
                
            #dnorm=Pnorm_bkgr[k]+MEFF
            dnorm=Pref
            E=0
            E = -log((P+MEFF)/dnorm)

            xs.append(d)
            ys.append(E)                
        #end for k
            
        #name="splines/%d_phi.txt"%(i+1)
        #out = open(name,'w')
        #out.write('x_axis'+'\t%7.2f'*len(xs)%tuple(xs)+'\n')
        #out.write('y_axis'+'\t%7.3f'*len(ys)%tuple(ys)+'\n')
        #out.close()

        #format: AtomPair CB 13 CB 37 SPLINE tag fname expt? weight binsize
        #form = 'Dihedral %s %d %s %d %s %d %s %d SPLINE TAG %s 1.0 1.0 0.1744 #0.0 0.0 0.0 %.3f\n'
        #j=i+1
        #if(j<nres): cstout.write(form%(atm1, j, atm2, j, atm3, j, atm4, j+1, name, weight))
    #end i
    #cstout.close()
    return top_phi, top_prob
#end rama_phi



def phi_cst(npz, ress,cstout):
    dat=npz['phi']
    #bkgr=npz['phi_bkgr']
    
    #cstout = file("cst_phi.txt", "w")

    for i,aa1 in enumerate(ress):

        atm1 = 'CA'
        atm2 = 'CB'

        if aa1 == 'GLY': continue
        
        for j,aa2 in enumerate(ress):            
            if(j == i): continue
            atm3 = 'CB'
            if aa2 == 'GLY': continue
            
            
            #check probability
            Praw = dat[i][j]
            #Praw_bkgr = bkgr[i][j]
            
            first_bin=1
            first_d=5
            
            Pnorm = [P for P in Praw[first_bin:]]            
            #Pnorm_bkgr=[P for P in Praw_bkgr[first_bin:]]

            xs = []
            ys = []
            weight=0
            for k,P in enumerate(Pnorm):weight += P
            #if weight<0.2: continue
            if(weight < PCUT1): continue

            Pref = Pnorm[-1]+MEFF #Reference probability at the last bin

            
            for k,P in enumerate(Pnorm):                            
                d = pid*(first_d + k*15)
                
                dnorm=Pref
                
                #dnorm=Pnorm_bkgr[k]+MEFF
                E = -log((P+MEFF)/dnorm)

                xs.append(d)
                ys.append(E)                
            #end for k
            name="splines/%d.%d_phi.txt"%(i+1,j+1)
            out = open(name,'w')
            out.write('x_axis'+'\t%7.2f'*len(xs)%tuple(xs)+'\n')
            out.write('y_axis'+'\t%7.3f'*len(ys)%tuple(ys)+'\n')
            out.close()
            #format: AtomPair CB 13 CB 37 SPLINE tag fname expt? weight binsize
            form = 'Angle %s %d %s %d %s %d SPLINE TAG %s 1.0 1.0 0.26 #0.0 0.0 0.0 %.3f\n'
            if(abs(i-j)>0): cstout.write(form%(atm1, i+1, atm2, i+1, atm3, j+1, name, weight))
        #end j
    #end i
    #cstout.close()
    
#end phi




def theta_cst(npz, ress,cstout):
    dat=npz['theta']
    #bkgr=npz['theta_bkgr']
    
    #cstout =file("cst_theta.txt", "w")

    for i,aa1 in enumerate(ress):

        atm1 = 'N'
        atm2 = 'CA'
        atm3 = 'CB'
        if aa1 == 'GLY': continue
        
        for j,aa2 in enumerate(ress):            
            if(j == i): continue
            atm4 = 'CB'
            if aa2 == 'GLY': continue
            
            
            #check probability
            Praw = dat[i][j]
            #Praw_bkgr = bkgr[i][j]
            
            first_bin=1
            first_d=-175
            
            Pnorm = [P for P in Praw[first_bin:]]            
            #Pnorm_bkgr=[P for P in Praw_bkgr[first_bin:]]

            xs = []
            ys = []
            weight=0
            for k,P in enumerate(Pnorm):weight += P
            #if weight<0.2: continue
            if(weight < PCUT1): continue

            Pref = Pnorm[-1]+MEFF
            for k,P in enumerate(Pnorm):                            
                d = pid*(first_d + k*15)
                
                #dnorm=Pnorm_bkgr[k]+MEFF
                dnorm=Pref
                E = -log((P+MEFF)/dnorm)

                xs.append(d)
                ys.append(E)                
            #end for k
            
            name="splines/%d.%d_theta.txt"%(i+1,j+1)
            out = open(name,'w')
            out.write('x_axis'+'\t%7.2f'*len(xs)%tuple(xs)+'\n')
            out.write('y_axis'+'\t%7.3f'*len(ys)%tuple(ys)+'\n')
            out.close()
            #format: AtomPair CB 13 CB 37 SPLINE tag fname expt? weight binsize
            form = 'Dihedral %s %d %s %d %s %d %s %d SPLINE TAG %s 1.0 1.0 0.26 #0.0 0.0 0.0 %.3f\n'
            if(abs(i-j)>0): cstout.write(form%(atm1, i+1, atm2, i+1, atm3, i+1, atm4, j+1, name, weight))
        #end j
    #end i
    #cstout.close()
    
#end theta




def omega_cst(npz, ress,cstout):
    dat=npz['omega']
    #bkgr=npz['omega_bkgr']
    
    #cstout = file("cst_omega.txt", "w")

    for i,aa1 in enumerate(ress):

        atm1 = 'CA'
        atm2 = 'CB'
        if aa1 == 'GLY': continue
        
        for j,aa2 in enumerate(ress):            
            if j < i+1: continue

            atm3 = 'CB'
            atm4 = 'CA'
            if aa2 == 'GLY': continue
            
            
            #check probability
            Praw = dat[i][j]
            #Praw_bkgr = bkgr[i][j]
            
            first_bin=1
            first_d=-180
            
            Pnorm = [P for P in Praw[first_bin:]]            
            #Pnorm_bkgr=[P for P in Praw_bkgr[first_bin:]]

            xs = []
            ys = []
            weight=0
            for k,P in enumerate(Pnorm):weight += P
            #if weight<0.2: continue
            if(weight < PCUT1): continue

            Pref = Pnorm[-1]+MEFF
            for k,P in enumerate(Pnorm):                            
                d = pid*(first_d + k*15)
                
                #dnorm=Pnorm_bkgr[k]+MEFF
                dnorm=Pref
                E = -log((P+MEFF)/dnorm)

                xs.append(d)
                ys.append(E)                
            #end for k
            
            name="splines/%d.%d_omega.txt"%(i+1,j+1)
            out = open(name,'w')
            out.write('x_axis'+'\t%7.2f'*len(xs)%tuple(xs)+'\n')
            out.write('y_axis'+'\t%7.3f'*len(ys)%tuple(ys)+'\n')
            out.close()
            #format: AtomPair CB 13 CB 37 SPLINE tag fname expt? weight binsize
            form = 'Dihedral %s %d %s %d %s %d %s %d SPLINE TAG %s 1.0 1.0 0.26 #0.0 0.0 0.0 %.3f\n'
            if(abs(i-j)>0): cstout.write(form%(atm1, i+1, atm2, i+1, atm3, j+1, atm4, j+1, name, weight))
        #end j
    #end i
    #cstout.close()
    
#end omega


def dist_cst(npz,ress,cstout, atm_type):

    if(atm_type=="CB"):
        dat=npz['dist']

    if(atm_type=="CA"):
        dat=npz['dist2']

    #bkgr=npz['dist_bkgr']

    aas = get_aa()
    pcut=PCUT
    rcut=PCUT
    #print pcut, rcut


    for i,aa1 in enumerate(ress):
        iaa1 = aas.index(aa1)
        atm1 = atm_type
        if(aa1 == 'GLY' and atm1 == "CB"): continue
        if(aa1 != 'GLY' and atm1 == "CA"): continue

        for j,aa2 in enumerate(ress):
            if j <= (i+0): continue

            #check probability
            Praw = dat[i][j]
            #Praw_bkgr = bkgr[i][j]

            #print(Praw[1:-1])
            #exit()
            atm2 = atm_type
            if(aa2 == 'GLY' and atm2 == "CB"): continue
            #if(aa2 != 'GLY' and atm2 == "CA"): continue
            iaa2 = aas.index(aa2)

            first_bin=4
            first_d=3.75 #4-->3.75, 5-->4.25
            weight=0
            plast=Praw[-1]
            for P in Praw[first_bin:]:
                if(P>bin_p):weight += P


            if(weight < pcut): continue


            Pnorm = [P for P in Praw[first_bin:]]
            #Pnorm_bkgr=[P for P in Praw_bkgr[first_bin:]]

            Pref = Pnorm[-1]+MEFF #Reference probability at the last bin

            probs=[]
            dists=[]
            xs = []
            ys = []
            dmax=0
            pmax=-1
            for k,P in enumerate(Pnorm):
                d = first_d + k*0.5
                if(P>pmax):

                    dmax=d
                    pmax=P
                #endif

                if(P>bin_p1):
                    probs.append(P)
                    dists.append(d)
                #endif

                dnorm = (d/DCUT)**ALPHA
                E = -log((P+MEFF)/(dnorm*Pref))

                #dnorm=1;
                E = -pow(P+MEFF, power)*log((P+MEFF)/(dnorm*Pref))

                #dnorm=Pnorm_bkgr[k]+MEFF
                #E = -log((P+MEFF)/dnorm)
                xs.append(d)
                ys.append(E)
            #end k

            e_dis=8;
            e_std=0;
            if(len(probs)==0):
                e_dis=dmax
                e_std=0;
            else:
                probs = [P/sum(probs) for P in probs]
                #print probs
                #print dists
                e_dis, e_std=weighted_avg_and_std(dists, probs)



            xs = [0.0, 2.0, 3]+xs

            ebase = EBASE *  pow(Pref, power)
            ys = [y + ebase for y in ys]
            y0 = max(ys[0], 0.0) #baseline of repulsion energy
            ys = [y0+EREP[0], y0+EREP[1], y0+EREP[2]] + ys #add repulsion on top of

            name="splines/%d.%d_%s.txt"%(i+1,j+1,atm_type)
            out = open(name,'w')
            out.write('x_axis'+'\t%7.2f'*len(xs)%tuple(xs)+'\n')
            out.write('y_axis'+'\t%7.3f'*len(ys)%tuple(ys)+'\n')
            out.close()
            #weight=1 #a constant 1 seems to be the best
            #weight1=weight
            #if((j-i)<=3): weight1=weight1 * 0.5
            #if((j-i)<=24 and (j-i)>3): weight1=weight1 * 0.7

            #format: AtomPair CB 13 CB 37 SPLINE tag fname expt? weight binsize
            form = 'AtomPair %4s %3d %4s %3d SPLINE TAG %s 1.0 %.3f 0.5 #%.3f %.3f %.3f %.3f\n'
            if(abs(i-j)>0): cstout.write(form%(atm1,i+1,atm2,j+1,name,1, e_dis, e_std, pmax, weight))
        #end j
    #end i
    #cstout.close()

#end dist_cst


def get_aa():
    return ['ALA','CYS','ASP','GLU','PHE',
            'GLY','HIS','ILE','LYS','LEU',
            'MET','ASN','PRO','GLN','ARG',
            'SER','THR','VAL','TRP','TYR']

def read_aas_pdb(pdb):
    aas = []
    for l in open(pdb):
        if not l.startswith('ATOM'): continue
        atm = l[12:16].strip()
        aa = l[16:20].strip()
        if atm == 'CA':
            aas.append(aa)
    return aas



def read_aas(filename):

    AA={
        'GLY':'G',
        'ALA':'A',
        'VAL':'V',
        'LEU':'L',
        'ILE':'I',
        'SER':'S',
        'THR':'T',
        'CYS':'C',
        'MET':'M',
        'PRO':'P',
        'ASP':'D',
        'ASN':'N',
        'GLU':'E',
        'GLN':'Q',
        'LYS':'K',
        'ARG':'R',
        'HIS':'H',
        'PHE':'F',
        'TYR':'Y',
        'TRP':'W',
        
        'G':'GLY',
        'A':'ALA',
        'V':'VAL',
        'L':'LEU',
        'I':'ILE',
        'S':'SER',
        'T':'THR',
        'C':'CYS',
        'M':'MET',
        'P':'PRO',
        'D':'ASP',
        'N':'ASN',
        'E':'GLU',
        'Q':'GLN',
        'K':'LYS',
        'R':'ARG',
        'H':'HIS',
        'F':'PHE',
        'Y':'TYR',
        'W':'TRP'
    }

    fasta="";
    with open(filename, "r") as f:
        for line in f:
            if(line[0] == ">"):
                continue
            else:
                line=line.rstrip()
                fasta = fasta + line;


    nres=len(fasta)
    seq=[]
    for i in range(0,nres):
        seq.append(AA[fasta[i]])
    return seq



def pairs_from_dpred(csts):
    pairs = []
    for cst in csts:
        for l in open(cst):
            wors = l[:-1].split()
            res1 = int(words[2]) 
            res2 = int(words[4])
            pairs.append((res1,res2))
    return pairs

def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    if(len(values)==1):return values[0],0

    average = np.average(values, weights=weights)
    variance=np.array(values).std()
    #v2=0
    #for w in weights:
    #    v2 += w**2
    #variance = np.average((values-average)**2, weights=weights)  # Fast and numerically precise
    #variance = sqrt(variance/(1-v2))
    return average, variance




if __name__ == '__main__':
    main()
