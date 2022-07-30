
### 1. 获取训练 pointwise listwise reranker 所需要的数据  
  
```bash  
python data/prepare_train_data.py 
```  
  
### 2. pointwise reranker 训练  

```bash  
sh run_train.sh
```  

### 3. 使用训练好的pointwise reranker进行去噪

```bash  
sh run_denoise_train.sh
```  

### 4. listwise reranker 训练  

```bash  
sh run_listwise_train.sh
```  

### 5. listwise + list-aware encoder reranker 训练  

```bash  
sh run_listwise_train_with_encoder.sh
```  

### 6. listwise reranker 预测  
  
```bash  
sh run_test.sh
``` 

### 7. listwise + list-aware encoder reranker 预测  
  
```bash  
sh run_test_with_encoder.sh
``` 

