#!/bin/bash

for i in $(seq 1 4)
do
python train_model.py S2648 50 120 5 0 5 1 0 0
python train_model.py S2648 50 120 5 0 5 2 0 0
done

for i in $(seq 1 4)
do
python train_model.py S2648 50 130 5 0 5 1 0 0
python train_model.py S2648 50 130 5 0 5 2 0 0
done

for i in $(seq 1 4)
do
python train_model.py S2648 50 140 5 0 5 1 0 0
python train_model.py S2648 50 140 5 0 5 2 0 0
done

for i in $(seq 1 4)
do
python train_model.py S2648 50 150 5 0 5 1 0 0
python train_model.py S2648 50 150 5 0 5 2 0 0
done

