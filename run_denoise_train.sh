

test_data=./data/train_data_top200
model_timestamp=bert_wwm_pointwise
model_type=bert_wwm
n_docs=200
file_name=train_pointwise_rerank


for n in $(seq 1 2);
 do
     if [ $n -eq 1 ]
         then
 	    python ${file_name}.py \
              --do_predict --valid_batch_size 128 \
              --test_data_path ${test_data}_0 \
              --gpu_id 0 \
              --n_docs ${n_docs} \
              --model_type ${model_type} \
              --model_timestamp ${model_timestamp} &
     elif [ $n -eq 2 ]
         then
 	    python ${file_name}.py \
              --do_predict --valid_batch_size 128 \
              --test_data_path ${test_data}_1 \
              --gpu_id 1 \
              --n_docs ${n_docs} \
              --model_type ${model_type} \
              --model_timestamp ${model_timestamp} &
     fi
 done
 wait

for n in $(seq 1 2);
 do
     if [ $n -eq 1 ]
         then
 	    python ${file_name}.py \
              --do_predict --valid_batch_size 128 \
              --test_data_path ${test_data}_2 \
              --gpu_id 0 \
              --n_docs ${n_docs} \
              --model_type ${model_type} \
              --model_timestamp ${model_timestamp} &
     elif [ $n -eq 2 ]
         then
 	    python ${file_name}.py \
              --do_predict --valid_batch_size 128 \
              --test_data_path ${test_data}_3 \
              --gpu_id 1 \
              --n_docs ${n_docs} \
              --model_type ${model_type} \
              --model_timestamp ${model_timestamp} &
     fi
 done
 wait