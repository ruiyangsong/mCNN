# classifier
time python classifier.py 120t atpe 100 0 full > classifier_mCNNwildCA120_atpe_eval100_testreportcla.log
time python classifier.py 120v atpe 100 0 full > classifier_mCNNwildCA120_atpe_eval100_valacc.log

# regressor
time python regressor.py 120t atpe 100 0 full > regressor_mCNNwildCA120_atpe_eval100_testreportreg.log
time python regressor.py 120v tpe 100 0 full > regressor_mCNNwildCA120_tpe_eval100_valmae.log

# multi_task
time python multi_task.py 120t tpe 100 0 full > multi_task_mCNNwildCA120_tpe_eval100_testreport.log
time python multi_task.py 120v tpe 100 0 full > multi_task_mCNNwildCA120_tpe_eval100_val.log