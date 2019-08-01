#!/bin/bash

for i in $(seq 1 5)
do
python train_model.py S2648 50 50 5 0 5 1 0 0
done

for i in $(seq 1 5)
do
python train_model.py S2648 50 50 5 0 5 1 1 0
done


for i in $(seq 1 5)
do
python train_model.py S2648 50 50 5 0 5 2 0 0
done

for i in $(seq 1 5)
do
python train_model.py S2648 50 50 5 0 5 2 1 0
done
