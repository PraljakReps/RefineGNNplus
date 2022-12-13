#!/usr/bin/env sh

python -V
export DIR="$(dirname "$(pwd)")"
#source activate torch_GPU
export PYTHONPATH=${PYTHONPATH}:${DIR}


export train_path='.././data/sabdab/hcdr3_cluster/train_data.jsonl'
export val_path='.././data/sabdab/hcdr3_cluster/val_data.jsonl'
export test_path='.././data/sabdab/hcdr3_cluster/test_data.jsonl'
export save_dir='../.ckpts/tmp'
export load_model=None
export output_path='.././output/hcdr1/sweep_nheads_hcdr1_RefineGNN+.csv'

# data and model hyperparameters
export hcdr=1
export hidden_size=256
export k_neighbors=9
export augment_eps=3.0
export depth=4
export vocab_size=21
export num_rbf=16
export dropout=0.1

# training hyperparameters
export lr=1e-3
export clip_norm=5.0
export epochs=5
export seed=7
export anneal_rate=0.9
export print_iter=50

# transformer hyperparameters
export nheads=8
export nheads_list='8'
export num_layers=3
export num_layers_list='3'
export search_optim='nheads'

python ../hp_optim_baseline_plus.py \
            --train_path ${train_path} \
            --val_path ${val_path} \
            --test_path ${test_path} \
            --epochs ${epochs} \
            --output_path ${output_path} \
            --hcdr ${hcdr} \
            --nheads ${nheads} \
            --nheads_list ${nheads_list} \
            --num_layers ${num_layers} \
            --num_layers_list ${num_layers_list} \
            --search_optim ${search_optim} \
            


