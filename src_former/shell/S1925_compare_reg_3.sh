#!/bin/bash

for i in $(seq 1 5)
do
python train_model.py S1925 13 0 5 0 20 1 1 0
python train_model.py S1925 13 0 5 0 20 2 1 0
done

for i in $(seq 1 5)
do
python train_model.py S1925 50 130 5 0 20 1 1 0
python train_model.py S1925 50 130 5 0 20 2 1 0
done

for i in $(seq 1 5)
do
python train_model.py S1925 14 0 5 0 20 1 1 0
python train_model.py S1925 14 0 5 0 20 2 1 0
done

for i in $(seq 1 5)
do
python train_model.py S1925 50 140 5 0 20 1 1 0
python train_model.py S1925 50 140 5 0 20 2 1 0
done

for i in $(seq 1 5)
do
python train_model.py S1925 15 0 5 0 20 1 1 0
python train_model.py S1925 15 0 5 0 20 2 1 0
done

for i in $(seq 1 5)
do
python train_model.py S1925 50 150 5 0 20 1 1 0
python train_model.py S1925 50 150 5 0 20 2 1 0
done
