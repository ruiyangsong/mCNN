#!/bin/bash

# model 1 for classification task that based on Datasets S2648.
# only include k_neighbor.

# input: dataset_name, radius, k_neighbor, class_num, k, nn_model, normalize_method, sort_method, (p_seed, k_seed, v_seed), (val_flag, verbose_flag)
for i in $(seq 1 5)
do
time python cross_validation1.py S2648 20 30 5 5 2.01 norm chain 1 1 1 0 0 0
done

for i in $(seq 1 5)
do
time python cross_validation1.py S2648 20 40 5 5 2.01 norm chain 1 1 1 0 0 0
done

for i in $(seq 1 5)
do
time python cross_validation1.py S2648 20 50 5 5 2.01 norm chain 1 1 1 0 0 0
done

for i in $(seq 1 5)
do
time python cross_validation1.py S2648 20 60 5 5 2.01 norm chain 1 1 1 0 0 0
done

for i in $(seq 1 5)
do
time python cross_validation1.py S2648 20 70 5 5 2.01 norm chain 1 1 1 0 0 0
done

for i in $(seq 1 5)
do
time python cross_validation1.py S2648 20 80 5 5 2.01 norm chain 1 1 1 0 0 0
done

for i in $(seq 1 5)
do
time python cross_validation1.py S2648 20 90 5 5 2.01 norm chain 1 1 1 0 0 0
done

for i in $(seq 1 5)
do
time python cross_validation1.py S2648 20 100 5 5 2.01 norm chain 1 1 1 0 0 0
done

for i in $(seq 1 5)
do
time python cross_validation1.py S2648 50 110 5 5 2.01 norm chain 1 1 1 0 0 0
done

for i in $(seq 1 5)
do
time python cross_validation1.py S2648 50 120 5 5 2.01 norm chain 1 1 1 0 0 0
done

for i in $(seq 1 5)
do
time python cross_validation1.py S2648 20 130 5 5 2.01 norm chain 1 1 1 0 0 0
done

for i in $(seq 1 5)
do
time python cross_validation1.py S2648 50 140 5 5 2.01 norm chain 1 1 1 0 0 0
done

for i in $(seq 1 5)
do
time python cross_validation1.py S2648 50 150 5 5 2.01 norm chain 1 1 1 0 0 0
done

for i in $(seq 1 5)
do
time python cross_validation1.py S2648 50 160 5 5 2.01 norm chain 1 1 1 0 0 0
done

for i in $(seq 1 5)
do
time python cross_validation1.py S2648 50 170 5 5 2.01 norm chain 1 1 1 0 0 0
done

for i in $(seq 1 5)
do
time python cross_validation1.py S2648 20 180 5 5 2.01 norm chain 1 1 1 0 0 0
done

for i in $(seq 1 5)
do
time python cross_validation1.py S2648 50 190 5 5 2.01 norm chain 1 1 1 0 0 0
done

for i in $(seq 1 5)
do
time python cross_validation1.py S2648 50 200 5 5 2.01 norm chain 1 1 1 0 0 0
done
