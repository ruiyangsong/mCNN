#!/usr/bin/env bash
# name_dataset, radius, k_neighbor, num_class

## generate all dataset_array of S2648.
# radius.
python calculate_neighbor.py S2648 3 0 5
python calculate_neighbor.py S2648 4 0 5
python calculate_neighbor.py S2648 5 0 5
python calculate_neighbor.py S2648 6 0 5
python calculate_neighbor.py S2648 7 0 5
python calculate_neighbor.py S2648 8 0 5
python calculate_neighbor.py S2648 9 0 5
python calculate_neighbor.py S2648 10 0 5
python calculate_neighbor.py S2648 11 0 5
python calculate_neighbor.py S2648 12 0 5
python calculate_neighbor.py S2648 13 0 5
python calculate_neighbor.py S2648 14 0 5
python calculate_neighbor.py S2648 15 0 5
# k_neighbor.
python calculate_neighbor.py S2648 20 30 5
python calculate_neighbor.py S2648 20 40 5
python calculate_neighbor.py S2648 20 50 5
python calculate_neighbor.py S2648 20 60 5
python calculate_neighbor.py S2648 20 70 5
python calculate_neighbor.py S2648 20 80 5
python calculate_neighbor.py S2648 20 90 5
python calculate_neighbor.py S2648 20 100 5
python calculate_neighbor.py S2648 50 110 5
python calculate_neighbor.py S2648 50 120 5
python calculate_neighbor.py S2648 20 130 5
python calculate_neighbor.py S2648 50 140 5
python calculate_neighbor.py S2648 50 150 5