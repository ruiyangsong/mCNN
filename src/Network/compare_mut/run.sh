#num=`ps aux | grep 20523 | wc -l`
#while [ ${num} -eq 2 ]
#do
#echo `date`', The former pid 20513 is still running...'
#sleep 300
#num=`ps aux | grep 20523 | wc -l`
#done

#for neighbor in {30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200}
for neighbor in {30}
do
echo -e '\n----------neighbor'$neighbor 'Begin neighbor at: ' `date`

echo -e '\n@classifier begin at: ' `date`
time python classifier.py $neighbor 3 full > classifier.log
echo -e '\n@classifier end at: ' `date`

echo -e '\n@regressor begin at: ' `date`
time python regressor.py $neighbor 3 full > regressor.log
echo -e '\n@regressor end at: ' `date`

echo -e '\n@multi_task begin at: ' `date`
time python multi_task.py $neighbor 3 full > multi_task.log
echo -e '\n@multi_task end at: ' `date`

echo -e '\n----------neighbor' $neighbor 'End at: ' `date`
done