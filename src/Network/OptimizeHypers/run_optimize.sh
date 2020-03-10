# classifier
date && time python classifier.py 120t atpe 100 0 full >> classifier_mCNNwildCA120_atpe_eval100_reduce5_testreportcla.log
date && time python classifier.py 120v atpe 100 0 full >> classifier_mCNNwildCA120_atpe_eval100_reduce5_valacc.log

# regressor
date && time python regressor.py 120t atpe 100 0 full >> regressor_mCNNwildCA120_atpe_eval100_reduce3_testreportreg.log
date && time python regressor.py 120v tpe 100 0 full >> regressor_mCNNwildCA120_tpe_eval100_reduce3_valmae.log

# multi_task
date && time python multi_task.py 120t tpe 100 0 full >> multi_task_mCNNwildCA120_tpe_eval100_reduce5_testreport.log
date && time python multi_task.py 120v tpe 100 0 full >> multi_task_mCNNwildCA120_tpe_eval100_reduce5_val.log