## classifier.py
### reduce=3
+ test_report_cla
    + 120t, atpe, eval=100, reduce_layer=3
+ val_acc
    + 120v, atpe, eval=100, reduce_layer=3 

## regresor
### reduce=3
+ test_report_reg
    + 120t, atpe, eval=100, reduce_layer=3
+ val_mae
    + 120v, **tpe**, eval=100, reduce_layer=3

## multi_task
### reduce=5
+ test_report
    + 120t, **tpe**, eval=100, **reduce_layer=5**
+ val
    + 120v, **tpe**, eval=100, **reduce_layer=5**
