echo '----------'
echo 'monitor val every 10 mins, begin at:'
echo `date`
echo '----------'
while true
do
date && cat multi_task_mCNNwildCA120_tpe_eval100_val.log | grep Best | wc -l
sleep 600
done
