#!/usr/bin/env python
'--sry,2020/3/12'
import os,time
def queueGPU(USER_MEM=10000,INTERVAL=600):
    try:
        totalmemlst=[int(x.split()[2]) for x in os.popen('nvidia-smi -q -d Memory |grep -A4 GPU|grep Total').readlines()]
        assert USER_MEM<=max(totalmemlst)
    except:
        print('\033[1;35m[WARNING]\nUSER_MEM should smaller than one of the GPU_TOTAL --> %s MiB.\nReset USER_MEM to %s MiB.\033[0m'%(totalmemlst,max(totalmemlst)-1))
        USER_MEM=max(totalmemlst)-1
    while True:
        memlst=[int(x.split()[2]) for x in os.popen('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free').readlines()]
        os.system("echo 'Check at:' `date`")
        print('GPU Free Memory List --> %s MiB\n'%memlst)
        idxlst=sorted(range(len(memlst)), key=lambda k: memlst[k])
        boollst=[y>USER_MEM for y in sorted(memlst)]
        try:
            GPU=idxlst[boollst.index(True)]
            os.environ['CUDA_VISIBLE_DEVICES']=str(GPU)
            print('GPU %s was chosen.\n'%GPU)
            break
        except:
            time.sleep(INTERVAL)
if __name__ == '__main__':
    queueGPU()
