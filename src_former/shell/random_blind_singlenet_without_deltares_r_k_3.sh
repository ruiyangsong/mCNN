#!/bin/bash

for i in $(seq 1 5)
do
python train_model.py S2648 10 0 5 0 1 1.03 0 0
python train_model.py S2648 10 0 5 0 1 2.03 0 0
done

for i in $(seq 1 5)
do
python train_model.py S2648 20 100 5 0 1 1.03 0 0
python train_model.py S2648 20 100 5 0 1 2.03 0 0
done

for i in $(seq 1 5)
do
python train_model.py S2648 11 0 5 0 1 1.03 0 0
python train_model.py S2648 11 0 5 0 1 2.03 0 0
done

for i in $(seq 1 5)
do
python train_model.py S2648 25 110 5 0 1 1.03 0 0
python train_model.py S2648 25 110 5 0 1 2.03 0 0
done
