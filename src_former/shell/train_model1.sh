#!/bin/bash

## full or part pca norm normalization

for i in $(seq 1 3)
do
echo "----- norm_single-net classification on S1925, octant_value = 1, k_neighbor = 50 -----"
time python train_model.py S1925 50 50 5 0 20 1 0 1
done

for i in $(seq 1 3)
do
echo "----- norm_single-net classification on S1925, octant_value = 1, k_neighbor = 60 -----"
time python train_model.py S1925 50 60 5 0 20 1 0 1
done

for i in $(seq 1 3)
do
echo "----- norm_single-net classification on S1925, octant_value = 1, k_neighbor = 70 -----"
time python train_model.py S1925 50 70 5 0 20 1 0 1
done

for i in $(seq 1 3)
do
echo "----- norm_single-net classification on S1925, octant_value = 1, k_neighbor = 80 -----"
time python train_model.py S1925 50 80 5 0 20 1 0 1
done

for i in $(seq 1 3)
do
echo "----- norm_single-net classification on S1925, octant_value = 1, k_neighbor = 90 -----"
time python train_model.py S1925 50 90 5 0 20 1 0 1
done

for i in $(seq 1 3)
do
echo "----- norm_single-net classification on S1925, octant_value = 1, k_neighbor = 100 -----"
time python train_model.py S1925 50 100 5 0 20 1 0 1
done

for i in $(seq 1 3)
do
echo "----- norm_single-net regression on S1925, octant_value = 1, k_neighbor = 50 -----"
time python train_model.py S1925 50 50 5 0 20 2 0 1
done

for i in $(seq 1 3)
do
echo "----- norm_single-net regression on S1925, octant_value = 1, k_neighbor = 60 -----"
time python train_model.py S1925 50 60 5 0 20 2 0 1
done

for i in $(seq 1 3)
do
echo "----- norm_single-net regression on S1925, octant_value = 1, k_neighbor = 70 -----"
time python train_model.py S1925 50 70 5 0 20 2 0 1
done

for i in $(seq 1 3)
do
echo "----- norm_single-net regression on S1925, octant_value = 1, k_neighbor = 80 -----"
time python train_model.py S1925 50 80 5 0 20 2 0 1
done

for i in $(seq 1 3)
do
echo "----- norm_single-net regression on S1925, octant_value = 1, k_neighbor = 90 -----"
time python train_model.py S1925 50 90 5 0 20 2 0 1
done

for i in $(seq 1 3)
do
echo "----- norm_single-net regression on S1925, octant_value = 1, k_neighbor = 100 -----"
time python train_model.py S1925 50 100 5 0 20 2 0 1
done