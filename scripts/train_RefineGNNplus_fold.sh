#!/usr/bin/env sh

python -V
export DIR="$(dirname "$(pwd)")"
#source activate torch_GPU
export PYTHONPATH=${PYTHONPATH}:${DIR}


export train_path='.././data/sabdab_2022_01/train_data.jsonl'
export val_path='.././data/sabdab_2022_01/val_data.jsonl'
export test_path='.././data/sabdab_2022_01/test_data.jsonl'
export save_dir='.././ckpts/RefineGNNplus-hfold/tmp'
export load_model=None
export output_path='.././output/structure_pred/train_model_cdr123.csv'

# data and model hyperparameters

export cdr=123
export hidden_size=256
export batch_tokens=200
export k_neighbors=9
export L_binder=150
export L_target=200
export depth=4
export rstep=4
export vocab_size=21
export num_rbf=16
export dropout=0.1

# training hyperparameter 
export lr=1e-3
export epochs=10
export seed=7
export print_iter=50
export anneal_rate=0.9
export clip_norm=1.0

# transformer hyperparameters
export nheads=8
export num_layers=8
export emb_dim=256

python ../fold_train_plus.py \
            --train_path ${train_path} \
            --val_path ${val_path} \
            --test_path ${test_path} \
            --save_dir ${save_dir} \
            --output_path ${output_path} \
            --epochs ${epochs} \
            --cdr ${cdr} \
            --nheads ${nheads} \
            --num_layers ${num_layers} \
            --emb_dim ${emb_dim} \

