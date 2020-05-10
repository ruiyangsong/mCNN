#[15, 12, 16, 29, 16, 12, 10, 31, 10, 19]
base_dir = '/dl/sry/mCNN/src/Network/deepddg/regressor/TrySimpleConv1D_CrossValid_2020.05.04.02.25.48'
best_epoch = []
for fold_num in range(1,11):
    history_dict_pth = '%s/fold_%s_history.dict'%(base_dir,fold_num)
    with open(history_dict_pth,'r') as f:
        history_dict = eval(f.read())
    val_loss = history_dict['val_loss']
    best_epoch.append(val_loss.index(min(val_loss)))
print(best_epoch)