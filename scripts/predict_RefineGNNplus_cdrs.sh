#!/usr/bin/env sh

python -V
export DIR="$(dirname "$(pwd)")"
#source activate torch_GPU
export PYTHONPATH=${PYTHONPATH}:${DIR}


export data_path='.././data/sabdab_2022_01/test_data.jsonl'
export save_dir='../RefineGNN+_pred_pdb/'
export load_model='.././ckpts/RefineGNNplus-hfold/best_model.ckpt.best'
export seed=7




python ../print_cdr_plus.py \
            --data_path ${data_path} \
            --save_dir ${save_dir} \
            --load_model ${load_model} \
            --seed ${seed} \



