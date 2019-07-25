import numpy as np
list_k=[130,140,150,160,170,180,190,200]
for k in list_k:
    max_dist=[]
    if not k in [50,120,130,140,150,160,170,180,190,200]:
        data=np.load('./S1925_r_20.00_neighbor_%d_class_5.npz'%k)
    else:
        data=np.load('./S1925_r_50.00_neighbor_%d_class_5.npz'%k)
    x=data['x']
    for i in range(x.shape[0]):
        max_disti = max(x[i,:,0])
        max_dist.append(max_disti)
    print('此数据集中最大的距离为:',max(max_dist))
    print('最小距离为：',min(max_dist))
    print('平均距离为:',np.mean(max_dist))
