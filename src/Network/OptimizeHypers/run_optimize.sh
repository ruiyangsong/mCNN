# classifier
time python classifier.py 120t atpe 100 0 full > classifier_mCNNwildCA120_atpe_eval100_testrepport.log
time python classifier.py 120v atpe 100 0 full > classifier_mCNNwildCA120_atpe_eval100_valacc.log

# regressor
time python regressor.py 120t atpe 100 0 full > regressor_mCNNwildCA120_atpe_eval100_testreportreg.log
time python regressor.py 120v atpe 100 0 full > regressor_mCNNwildCA120_atpe_eval100_valmae.log

# multi_task
