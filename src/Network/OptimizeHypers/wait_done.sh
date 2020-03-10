num=`ps aux | grep 29783 | wc -l`
while [ ${num} -eq 2 ]
do
echo `date`', The former shell is still running...'
sleep 300
num=`ps aux | grep 29783 | wc -l`
done

echo '---Begin multi_task at: ' `date`
time python multi_task.py 120t tpe 100 0 full > multi_task_mCNNwildCA120_tpe_eval100_testreport.log
time python multi_task.py 120v tpe 100 0 full > multi_task_mCNNwildCA120_tpe_eval100_val.log
echo '---End multi_task at: ' `date`
