#!/bin/bash


for i in $(seq 1 5)
do
python train_model.py S2648 50 130 5 0 5 1 1 0
python train_model.py S2648 50 130 5 0 5 2 1 0
done

for i in $(seq 1 5)
do
python train_model.py S2648 50 140 5 0 5 1 1 0
python train_model.py S2648 50 140 5 0 5 2 1 0
done

for i in $(seq 1 5)
do
python train_model.py S2648 50 150 5 0 5 1 1 0
python train_model.py S2648 50 150 5 0 5 2 1 0
done

for i in $(seq 1 5)
do
python train_model.py S2648 13 0 5 0 5 1 1 0
python train_model.py S2648 13 0 5 0 5 2 1 0
done

for i in $(seq 1 5)
do
python train_model.py S2648 14 0 5 0 5 1 1 0
python train_model.py S2648 14 0 5 0 5 2 1 0
done

for i in $(seq 1 5)
do
python train_model.py S2648 15 0 5 0 5 1 1 0
python train_model.py S2648 15 0 5 0 5 2 1 0
done
