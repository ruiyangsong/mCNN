echo 'task 1 begin at:' `date`
time python ./classifier/reduce5/classifier.py 120t atpe 100 0 full >> ./classifier/reduce5/classifier_mCNNwildCA120_atpe_eval100_reduce5_testreportcla.log
echo 'task 1 end at:' `date`

echo 'task 2 begin at:' `date`
time python ./classifier/reduce5/classifier.py 120v atpe 100 0 full >> ./classifier/reduce5/classifier_mCNNwildCA120_atpe_eval100_reduce5_valacc.log
echo 'task 2 end at:' `date`