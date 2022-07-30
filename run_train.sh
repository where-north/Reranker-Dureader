

python train_pointwise_rerank.py \
--do_train \
--epoch 2 \
--lr 1e-5 \
--model_type bert_wwm \
--batch_size 16 \
--q_maxlen 32 \
--p_head_maxlen 384 \
--p_tail_maxlen 0 \
--train_data_path ./data/pointwise_train_data/reranker_train.tsv \
--valid_data_path ./data/pointwise_train_data/reranker_valid.tsv \
--wise pointwise \
--gpu_id 0 \
--fp16 \
--do_adversarial \
--fgm_epsilon 0.01


