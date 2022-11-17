rm -rf ensemble/main_reinforced_gnn_ncf

python3 main_reinforced_gnn_ncf.py --ensemble-learning --random-seed 7
python3 main_reinforced_gnn_ncf.py --ensemble-learning --random-seed 27
python3 main_reinforced_gnn_ncf.py --ensemble-learning --random-seed 310
python3 main_reinforced_gnn_ncf.py --ensemble-learning --random-seed 128
python3 main_reinforced_gnn_ncf.py --ensemble-learning --random-seed 841
python3 main_reinforced_gnn_ncf.py --ensemble-learning --random-seed 99
python3 main_reinforced_gnn_ncf.py --ensemble-learning --random-seed 707
python3 main_reinforced_gnn_ncf.py --ensemble-learning --random-seed 505
python3 main_reinforced_gnn_ncf.py --ensemble-learning --random-seed 606
python3 main_reinforced_gnn_ncf.py --ensemble-learning --random-seed 808
python3 main_reinforced_gnn_ncf.py --ensemble-learning --random-seed 999
python3 main_reinforced_gnn_ncf.py --ensemble-learning --random-seed 7777

python3 ../utilities/get_mean_predictions.py --ensemble-directory ensemble/reinforced_gnn_ncf