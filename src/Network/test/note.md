test/test1 test/test2 test/test3 三个文件夹进行了三组分类实验的超参数优化，基于[center_CA, classifier]
## 第一组:
    训练集:wild的80%和所有mutant
    测试集:wild的20%
## 第二组:
    训练集:wild的80%和所有mutant
    测试集:wild的20%
## 第三组:
    训练集:mutant的80%和所有mutant
    测试集:mutant的20%
**每一组都选用参数优化,然后对比效果,若第二组好,则训练卷积基时选择mutant数据**

调节超参数的两种思路：
1. 每一折都调节，最后综合确定（如果k组参数差别很大，说明模型过拟合，应重新切分验证集试试，**多次都不行的话目前没有想法**）
2. 先确定超参数再进行k折CV（这样的话相当于k折中的测试集用到了调参数中）
