# 分类
## classifier/reduce3
优化目标
+ test_report_cla
    + 120t, atpe, eval=100:wq
    :
+ val_acc
    + 120v, atpe, eval=100
    
## classifier/reduce5
优化目标
+ test_report_cla
    + 120t, atpe, eval=100
此时发现以test_report为优化目标效果不如**val_acc**，停止了后续的reduce5实验。

## classifier/reduceAUTO
***

# 回归
## regresor/reduce3
优化目标
+ test_report_reg
    + 120t, atpe, eval=100
+ val_mae
    + 120v, **tpe**, eval=100
最优参数的回归效果很差，可能网络结构有问题（参数过少？），没有进行后续的reduce5实验

## regresor/reduceAUTO

***

# 多任务
## multi_task/reduce5
+ test_report
    + 120t, **tpe**, eval=100
+ val
    + 120v, **tpe**, eval=100

## multi_task/reduceAUTO