#!/bin/bash


for i in $(seq 1 5)
do
python train_model.py S2648 50 50 5 0 5 1.03 0 0
python train_model.py S2648 50 50 5 0 5 2.03 0 0
done

for i in $(seq 1 5)
do
python train_model.py S2648 20 70 5 0 5 1.03 0 0
python train_model.py S2648 20 70 5 0 5 2.03 0 0
done

for i in $(seq 1 5)
do
python train_model.py S2648 20 100 5 0 5 1.03 0 0
python train_model.py S2648 20 100 5 0 5 2.03 0 0
done

