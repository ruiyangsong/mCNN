#!/bin/bash

for i in $(seq 1 5)
do
python train_model.py S2648 3 0 5 0 1 1 0 0
python train_model.py S2648 3 0 5 0 1 2 0 0
done

for i in $(seq 1 5)
do
python train_model.py S2648 20 30 5 0 1 1 0 0
python train_model.py S2648 20 30 5 0 1 2 0 0
done

for i in $(seq 1 5)
do
python train_model.py S2648 4 0 5 0 1 1 0 0
python train_model.py S2648 4 0 5 0 1 2 0 0
done

for i in $(seq 1 5)
do
python train_model.py S2648 20 40 5 0 1 1 0 0
python train_model.py S2648 20 40 5 0 1 2 0 0
done

for i in $(seq 1 5)
do
python train_model.py S2648 5 0 5 0 1 1 0 0
python train_model.py S2648 5 0 5 0 1 2 0 0
done

for i in $(seq 1 5)
do
python train_model.py S2648 50 50 5 0 1 1 0 0
python train_model.py S2648 50 50 5 0 1 2 0 0
done


for i in $(seq 1 5)
do
python train_model.py S2648 6 0 5 0 1 1 0 0
python train_model.py S2648 6 0 5 0 1 2 0 0
done

for i in $(seq 1 5)
do
python train_model.py S2648 20 60 5 0 1 1 0 0
python train_model.py S2648 20 60 5 0 1 2 0 0
done
