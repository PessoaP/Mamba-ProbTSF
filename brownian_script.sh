export CUDA_VISIBLE_DEVICES=0
model_name=S_Mamba

python -u run_prob.py \
  --is_training 1 \
  --root_path ./dataset/Brownian/ \
  --data_path brownian.csv \
  --model_id Brownian_128_128 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 128 \
  --pred_len 128 \
  --e_layers 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --d_state 16 \
  --train_epochs 5 \
  --batch_size 16 \
  --learning_rate 0.001 \
  --itr 1 \
  --sigma_network Mamba \
  --sigma_method General \
 
python -u run_prob.py \
  --is_training 1 \
  --root_path ./dataset/Brownian/ \
  --data_path brownian.csv \
  --model_id Brownian_Linear_128_128 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 128 \
  --pred_len 128 \
  --e_layers 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --d_state 16 \
  --train_epochs 5 \
  --batch_size 16 \
  --learning_rate 0.001 \
  --itr 1 \
  --sigma_network Linear \
  --sigma_method General \


