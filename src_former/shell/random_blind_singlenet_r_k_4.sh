#!/bin/bash

for i in $(seq 1 5)
do
python train_model.py S2648 50 120 5 0 1 1 0 0
python train_model.py S2648 50 120 5 0 1 2 0 0
done

for i in $(seq 1 5)
do
python train_model.py S2648 12 0 5 0 1 1 0 0
python train_model.py S2648 12 0 5 0 1 2 0 0
done

for i in $(seq 1 5)
do
python train_model.py S2648 13 0 5 0 1 1 0 0
python train_model.py S2648 13 0 5 0 1 2 0 0
done

for i in $(seq 1 5)
do
python train_model.py S2648 14 0 5 0 1 1 0 0
python train_model.py S2648 14 0 5 0 1 2 0 0
done

for i in $(seq 1 5)
do
python train_model.py S2648 15 0 5 0 1 1 0 0
python train_model.py S2648 15 0 5 0 1 2 0 0
done
