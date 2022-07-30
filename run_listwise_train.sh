

python train_listwise_rerank.py \
--do_train \
--epoch 2 \
--lr 5e-6 \
--model_type nezha_base \
--batch_size 2 \
--valid_batch_size 50 \
--q_maxlen 32 \
--p_head_maxlen 384 \
--p_tail_maxlen 0 \
--train_data_path ./data/listwise_train_data/25_samples_seed1234/reranker_train.json \
--valid_data_path ./data/listwise_train_data/reranker_valid.json \
--wise listwise \
--train_group_size 26 \
--valid_n_docs 50 \
--gpu_id 1 \
--fp16 \
--warmup \
--warmup_ratio 0.1 \
--accumulation_steps 1 \
--seed 42 \
--start_save_steps 30000 \
--save_steps 2000 \
#--do_adversarial \
#--fgm_epsilon 0.01

#python train_pointwise_rerank.py \
#--do_train \
#--epoch 3 \
#--gpu_id 3

