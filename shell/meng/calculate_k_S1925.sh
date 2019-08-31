#!/usr/bin/env bash
# name_dataset, radius, k_neighbor, num_class

## generate only k_neighbor dataset_array of S1925.
# k_neighbor.
python calculate_neighbor.py S1925 50 30 5
python calculate_neighbor.py S1925 50 40 5
python calculate_neighbor.py S1925 50 50 5
python calculate_neighbor.py S1925 50 60 5
python calculate_neighbor.py S1925 50 70 5
python calculate_neighbor.py S1925 50 80 5
python calculate_neighbor.py S1925 50 90 5
python calculate_neighbor.py S1925 50 100 5
python calculate_neighbor.py S1925 50 110 5
python calculate_neighbor.py S1925 50 120 5