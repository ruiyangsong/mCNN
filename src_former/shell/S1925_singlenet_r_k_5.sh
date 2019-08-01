#!/bin/bash

for i in $(seq 1 5)
do
python train_model.py S1925 15 0 5 0 20 1 0 0
python train_model.py S1925 15 0 5 0 20 2 0 0
done

for i in $(seq 1 5)
do
python train_model.py S1925 50 150 5 0 20 1 0 0
python train_model.py S1925 50 150 5 0 20 2 0 0
done
