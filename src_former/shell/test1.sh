#!/bin/bash

for i in $(seq 1 5)
do
python train_model.py S2648 50 50 5 0 5 2.02 0 1
done
