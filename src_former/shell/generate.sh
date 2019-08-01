#!/bin/bash

# generate datasets_array

python generate_dataset.py S2648 12 0 5 && python calculate_neighbor.py S2648 13 0 && python generate_dataset.py S2648 13 0 5 && python calculate_neighbor.py S2648 14 0 && python generate_dataset.py S2648 14 0 5 && python calculate_neighbor.py S2648 15 0 && python generate_dataset.py S2648 15 0 5 
# echo "1" && echo "2" && echo "3"
