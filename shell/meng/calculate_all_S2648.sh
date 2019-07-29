#!/usr/bin/env bash
# name_dataset, radius, k_neighbor, num_class
# generate all dataset_array of S2648 and only k_neighbor dataset_array of S1925.
python src/calculate_neighbor.py S2648 3 0 5
python src/calculate_neighbor.py S2648 4 0 5
python src/calculate_neighbor.py S2648 5 0 5
python src/calculate_neighbor.py S2648 6 0 5
python src/calculate_neighbor.py S2648 7 0 5
python src/calculate_neighbor.py S2648 8 0 5
python src/calculate_neighbor.py S2648 9 0 5
python src/calculate_neighbor.py S2648 10 0 5

python src/calculate_neighbor.py S2648 20 30 5
python src/calculate_neighbor.py S2648 20 40 5
python src/calculate_neighbor.py S2648 20 50 5
python src/calculate_neighbor.py S2648 20 60 5
python src/calculate_neighbor.py S2648 20 70 5
python src/calculate_neighbor.py S2648 20 80 5
python src/calculate_neighbor.py S2648 20 90 5
python src/calculate_neighbor.py S2648 20 100 5
python src/calculate_neighbor.py S2648 50 110 5
python src/calculate_neighbor.py S2648 50 120 5