export CUDA_VISIBLE_DEVICES=0

model_name=TimeLLM
train_epochs=50
llama_layers=32

master_port=10097
num_process=1
batch_size=(16)
patch_lens=(1)
strides=(1)
d_models=(32) #dim
d_ffs=(128)
learning_rates=(0.001)
num_tokens_s=(200)
is_training=1
comment='TimeLLM-BP'
load_model=0

for stride in "${strides[@]}"
do
  for patch_len in "${patch_lens[@]}"
  do
    for d_model in "${d_models[@]}"
    do
      for d_ff in "${d_ffs[@]}"
      do
        for num_tokens in "${num_tokens_s[@]}"
        do
          for learning_rate in "${learning_rates[@]}"
            do
              echo "Run with batch_size $batch_size, d_model $d_model, d_ff $d_ff, learning_rate $learning_rate, num_tokens $num_tokens, patch_len $patch_len, stride $stride"
              accelerate launch --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main_BP.py \
                --M_phi 0 \
                --checkpoints ./checkpoints/ \
                --speeds 10 15 20 \
                --num_antenna 32 64 128 \
                --patience 5 \
                --islora 0 \
                --task_name long_term_forecast \
                --is_training $is_training \
                --root_path ./dataset/BP_dataset/ \
                --model_id BP_32_16 \
                --model $model_name \
                --data BP \
                --features MS \
                --seq_len 40 \
                --pred_len 10 \
                --factor 3 \
                --enc_in 1 \
                --dec_in 1 \
                --c_out 1 \
                --patch_len $patch_len \
                --stride $stride \
                --des 'Exp' \
                --itr 1 \
                --num_tokens $num_tokens \
                --d_model $d_model \
                --d_ff $d_ff \
                --batch_size $batch_size \
                --learning_rate $learning_rate \
                --llm_layers $llama_layers \
                --train_epochs $train_epochs \
                --model_comment $comment \
                --load_model $load_model \
                --llm_model GPT2 \
                --llm_dim 1280
            done
          done
      done
    done
  done
done










