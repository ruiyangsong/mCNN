#!/bin/bash

for i in $(seq 1 4)
do
python train_model.py S2648 7 0 5 0 5 1.03 0 0
python train_model.py S2648 7 0 5 0 5 2.03 0 0
done

for i in $(seq 1 4)
do
python train_model.py S2648 20 70 5 0 5 1.03 0 0
python train_model.py S2648 20 70 5 0 5 2.03 0 0
done

for i in $(seq 1 4)
do
python train_model.py S2648 8 0 5 0 5 1.03 0 0
python train_model.py S2648 8 0 5 0 5 2.03 0 0
done

for i in $(seq 1 4)
do
python train_model.py S2648 20 80 5 0 5 1.03 0 0
python train_model.py S2648 20 80 5 0 5 2.03 0 0
done

for i in $(seq 1 4)
do
python train_model.py S2648 9 0 5 0 5 1.03 0 0
python train_model.py S2648 9 0 5 0 5 2.03 0 0
done

for i in $(seq 1 4)
do
python train_model.py S2648 20 90 5 0 5 1.03 0 0
python train_model.py S2648 20 90 5 0 5 2.03 0 0
done
