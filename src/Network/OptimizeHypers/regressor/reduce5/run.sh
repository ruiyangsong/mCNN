echo 'task 1 begin at:' `date`
time python ./regressor/reduce5/regressor.py 120t atpe 100 0 full >> ./regressor/reduce5/regressor_mCNNwildCA120_atpe_eval100_reduce5_testreportreg.log
echo 'task 1 end at:' `date`

echo 'task 2 begin at:' `date`
time python ./regressor/reduce5/regressor.py 120v tpe 100 0 full >> ./regressor/reduce5/regressor_mCNNwildCA120_tpe_eval100_reduce5_valmae.log
echo 'task 2 end at:' `date`