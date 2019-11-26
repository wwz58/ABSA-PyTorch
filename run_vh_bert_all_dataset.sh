CUDA_VISIBLE_DEVICES=0 python main.py --do_train \
    --do_eval \
    --model_name vh_bert \
    --dataset laptop \
    --exp_id 0 &

CUDA_VISIBLE_DEVICES=1 python main.py --do_train \
    --do_eval \
    --model_name vh_bert \
    --dataset restaurant \
    --exp_id 0 &

wait
CUDA_VISIBLE_DEVICES=0 python main.py --do_train \
    --do_eval \
    --model_name vh_bert \
    --dataset twitter \
    --exp_id 0 
