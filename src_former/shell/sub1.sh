line=`ps aux|grep 27027|grep -v "grep"|wc -l`
while [ $line -ge 1 ]
do
	echo $(date)
	sleep 1h
	line=`ps aux|grep 4151|grep -v "grep"|wc -l`
done
nohup bash shell/compare_reg_4.sh > results/compare_reg_4.log 2>&1 &
