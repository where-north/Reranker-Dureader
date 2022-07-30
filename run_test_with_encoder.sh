

test_data=./data/no_denoise/reranker_test_50.json
model_timestamp=2022-06-24_09_23_09
model_type=macbert_base
n_docs=50
p_head_maxlen=384
p_tail_maxlen=0
valid_batch_size=50
file_name=train_listwise_rerank


#for n in $(seq 1 2);
# do
#     if [ $n -eq 1 ]
#         then
# 	    python ${file_name}.py \
#              --do_predict_with_encoder --valid_batch_size ${valid_batch_size} \
#              --test_data_path ${test_data}_0 \
#              --q_maxlen 32 \
#              --p_head_maxlen ${p_head_maxlen} \
#              --p_tail_maxlen ${p_tail_maxlen} \
#              --gpu_id 1 \
#              --n_docs ${n_docs} \
#              --model_type ${model_type} \
#              --model_timestamp ${model_timestamp} &
#     elif [ $n -eq 2 ]
#         then
# 	    python ${file_name}.py \
#              --do_predict_with_encoder --valid_batch_size ${valid_batch_size} \
#              --test_data_path ${test_data}_1 \
#              --q_maxlen 32 \
#              --p_head_maxlen ${p_head_maxlen} \
#              --p_tail_maxlen ${p_tail_maxlen} \
#              --gpu_id 3 \
#              --n_docs ${n_docs} \
#              --model_type ${model_type} \
#              --model_timestamp ${model_timestamp} &
#     fi
# done
# wait
#
#for n in $(seq 1 2);
# do
#     if [ $n -eq 1 ]
#         then
# 	    python ${file_name}.py \
#              --do_predict_with_encoder --valid_batch_size ${valid_batch_size} \
#              --test_data_path ${test_data}_2 \
#              --q_maxlen 32 \
#              --p_head_maxlen ${p_head_maxlen} \
#              --p_tail_maxlen ${p_tail_maxlen} \
#              --gpu_id 1 \
#              --n_docs ${n_docs} \
#              --model_type ${model_type} \
#              --model_timestamp ${model_timestamp} &
#     elif [ $n -eq 2 ]
#         then
# 	    python ${file_name}.py \
#              --do_predict_with_encoder --valid_batch_size ${valid_batch_size} \
#              --test_data_path ${test_data}_3 \
#              --q_maxlen 32 \
#              --p_head_maxlen ${p_head_maxlen} \
#              --p_tail_maxlen ${p_tail_maxlen} \
#              --gpu_id 3 \
#              --n_docs ${n_docs} \
#              --model_type ${model_type} \
#              --model_timestamp ${model_timestamp} &
#     fi
# done
# wait

for n in $(seq 1 4);
 do
     if [ $n -eq 1 ]
         then
 	    python ${file_name}.py \
              --do_predict_with_encoder --valid_batch_size ${valid_batch_size} \
              --test_data_path ${test_data}_0 \
              --q_maxlen 32 \
              --p_head_maxlen ${p_head_maxlen} \
              --p_tail_maxlen ${p_tail_maxlen} \
              --gpu_id 0 \
              --n_docs ${n_docs} \
              --model_type ${model_type} \
              --model_timestamp ${model_timestamp} &
     elif [ $n -eq 2 ]
         then
 	    python ${file_name}.py \
              --do_predict_with_encoder --valid_batch_size ${valid_batch_size} \
              --test_data_path ${test_data}_1 \
              --q_maxlen 32 \
              --p_head_maxlen ${p_head_maxlen} \
              --p_tail_maxlen ${p_tail_maxlen} \
              --gpu_id 1 \
              --n_docs ${n_docs} \
              --model_type ${model_type} \
              --model_timestamp ${model_timestamp} &

     elif [ $n -eq 3 ]
         then
 	    python ${file_name}.py \
              --do_predict_with_encoder --valid_batch_size ${valid_batch_size} \
              --test_data_path ${test_data}_2 \
              --q_maxlen 32 \
              --p_head_maxlen ${p_head_maxlen} \
              --p_tail_maxlen ${p_tail_maxlen} \
              --gpu_id 2 \
              --n_docs ${n_docs} \
              --model_type ${model_type} \
              --model_timestamp ${model_timestamp} &
     elif [ $n -eq 4 ]
         then
 	    python ${file_name}.py \
              --do_predict_with_encoder --valid_batch_size ${valid_batch_size} \
              --test_data_path ${test_data}_3 \
              --q_maxlen 32 \
              --p_head_maxlen ${p_head_maxlen} \
              --p_tail_maxlen ${p_tail_maxlen} \
              --gpu_id 3 \
              --n_docs ${n_docs} \
              --model_type ${model_type} \
              --model_timestamp ${model_timestamp} &
     fi
 done
wait