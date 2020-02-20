# Notice
* ref.py 操作后得到的pdb文件中的HEADER部分中的 ss_bond 信息会影响到后续 mut.py 的 mutant 操作 (Rosetta建立的结构可能与 ss_bond 产生冲突)

# To Do List
## stage 1
***
* Data 中加入normalization
***
* 开始写 cross_validation
## stage 2
* 若 append mCSM to mCNN 检查 mCNN 和 mCSM 中每一个样本的顺序是否对应！(尽量不用这种拼接方法)
* CNN 加入残差
* Res2Net 实验

# code structure
* 数据输入网络可能的形式{mCNN:[wild_only,mutant_only,stack,split],mCSM:[wild_only,mutant_only,stack,split]}


# code note
## plot model
nn_model, sample_size = 3, (50,59)
model = build_model(nn_model, sample_size)
from IPython.display import SVG, display
from keras.utils.vis_utils import model_to_dot
/# plot_model(model,show_shapes=True,to_file='model1.png')
/# plot_model(model, show_shapes=True, to_file='model.png'
