#!/bin/bash

## max normalization

for i in $(seq 1 3)
do
echo "-------------------- single-net classification on S1925 --------------------"
time python train_model1.py S1925 50 50 5 0 20 1 1
done
for i in $(seq 1 3)
do
echo "-------------------- multi-net classification on S1925 --------------------"
time python train_model1.py S1925 50 120 5 0 20 1.01 1
done
for i in $(seq 1 3)
do
echo "-------------------- single-net regression on S1925 --------------------"
time python train_model1.py S1925 50 50 5 0 20 2 1
done
for i in $(seq 1 3)
do
echo "-------------------- multi-net regression on S1925 --------------------"
time python train_model1.py S1925 50 120 5 0 20 2.01 1
done

for i in $(seq 1 3)
do
echo "-------------------- single-net classification on S2648 --------------------"
time python train_model1.py S2648 50 50 5 0 5 1 1
done
for i in $(seq 1 3)
do
echo "-------------------- multi-net classification on S2648 --------------------"
time python train_model1.py S2648 50 120 5 0 5 1.01 1
done
for i in $(seq 1 3)
do
echo "-------------------- single-net regression on S2648 --------------------"
time python train_model1.py S2648 50 50 5 0 5 2 1
done
for i in $(seq 1 3)
do
echo "-------------------- multi-net regression on S2648 --------------------"
time python train_model1.py S2648 50 120 5 0 5 2.01 1
done