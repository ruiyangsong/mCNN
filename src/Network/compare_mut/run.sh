#num=`ps aux | grep 20523 | wc -l`
#while [ ${num} -eq 2 ]
#do
#echo `date`', The former pid 20523 is still running...'
#sleep 300
#num=`ps aux | grep 20523 | wc -l`
#done

/dl/sry/bin/queueGPU
$CUDA_RATE=$1
lst="30 40 50 60 70 80 90 100 110 120 130 140 150 160 170 180 190 200"
for neighbor in $lst
do
echo -e '\n----------neighbor' $neighbor 'Begin at: ' `date`

echo -e '\n@classifier begin at: ' `date`
time python classifier.py $neighbor $CUDA_RATE >> classifier.log
time python classifier.py $neighbor $CUDA_RATE >> classifier.log
echo -e '\n@classifier end at: ' `date`

echo -e '\n@regressor begin at: ' `date`
time python regressor.py $neighbor full >> regressor.log
time python regressor.py $neighbor $CUDA_RATE >> regressor.log
echo -e '\n@regressor end at: ' `date`

echo -e '\n@multi_task begin at: ' `date`
time python multi_task.py $neighbor $CUDA_RATE >> multi_task.log
time python multi_task.py $neighbor $CUDA_RATE >> multi_task.log
echo -e '\n@multi_task end at: ' `date`

echo -e '\n----------neighbor' $neighbor 'End at: ' `date`
done


sleep 5
kill -CONT 20521
sleep 10
kill -CONT 20518