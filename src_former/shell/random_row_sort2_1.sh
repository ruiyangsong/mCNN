#!/bin/bash

for i in $(seq 1 5)
do
python train_model.py S2648 20 100 5 0 5 1.02 0 0
python train_model.py S2648 20 100 5 0 5 2.02 0 0
done
