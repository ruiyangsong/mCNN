echo '----------'
echo 'monitor testreportreg every 10 mins, begin at:'
echo `date`
echo '----------'
while true
do
date && cat regressor_mCNNwildCA120_atpe_eval100_testreportreg.log | grep Predict | wc -l
sleep 600
done
