echo '----------'
echo 'monitor testreport every 10 mins, begin at:'
echo `date`
echo '----------'
while true
do
date && cat multi_task_mCNNwildCA120_tpe_eval100_testreport.log | grep Predict | wc -l
sleep 600
done
