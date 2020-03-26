CUDA_RATE=$1
#date && time python regressor.py 120t atpe 100 0 $CUDA_RATE > regressor_mCNNwildCA120_atpe_eval100_reduce3_testreportreg.log
date && time python regressor.py 120v tpe 100 0 $CUDA_RATE > regressor_mCNNwildCA120_tpe_eval100_reduce3_valmae.log
