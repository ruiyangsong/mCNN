#!/bin/bash

for i in $(seq 1 2)
do
python train_model.py S2648 50 120 5 0 5 2.01 0 0
done
