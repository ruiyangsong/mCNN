#!/bin/bash

for i in $(seq 1 5)
do
python train_model.py S1925 50 120 5 0 20 1.03 0 0
python train_model.py S1925 50 120 5 0 20 2.03 0 0
done

for i in $(seq 1 5)
do
python train_model.py S1925 12 0 5 0 20 1.03 0 0
python train_model.py S1925 12 0 5 0 20 2.03 0 0
done

for i in $(seq 1 5)
do
python train_model.py S1925 50 130 5 0 20 1.03 0 0
python train_model.py S1925 50 130 5 0 20 2.03 0 0
done


for i in $(seq 1 5)
do
python train_model.py S1925 13 0 5 0 20 1.03 0 0
python train_model.py S1925 13 0 5 0 20 2.03 0 0
done

#for i in $(seq 1 5)
#do
#python train_model.py S2648 14 0 5 0 5 1 0 0
#python train_model.py S2648 14 0 5 0 5 2 0 0
#done

#for i in $(seq 1 5)
#do
#python train_model.py S2648 15 0 5 0 5 1 0 0
#python train_model.py S2648 15 0 5 0 5 2 0 0
#done
