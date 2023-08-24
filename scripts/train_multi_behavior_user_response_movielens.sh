mkdir -p output

mkdir -p output/ml1m/
mkdir -p output/ml1m/env
mkdir -p output/ml1m/env/log

data_path='../dataset/ml1m/'
train_path=${data_path}log_session.csv
user_meta_path=${data_path}users_processed.csv
item_meta_path=${data_path}movies_processed.csv
output_path='output/ml1m/'

# MODEL='KRMBUserResponse'
MODEL='KRMBUserResponse_MaxOut'

for LR in 0.0001 0.00001 0.001
do
    for REG in 0.01 0.1 0
    do
        python train_multibehavior.py\
            --epoch 10\
            --seed 19\
            --lr ${LR}\
            --batch_size 128\
            --val_batch_size 128\
            --cuda 0\
            --reader MLSeqReader\
            --train_file ${train_path}\
            --user_meta_file ${user_meta_path}\
            --item_meta_file ${item_meta_path}\
            --max_hist_seq_len 50\
            --data_separator ','\
            --meta_file_separator ','\
            --n_worker 4\
            --val_holdout_per_user 5\
            --test_holdout_per_user 5\
            --model ${MODEL}\
            --loss 'bce'\
            --l2_coef ${REG}\
            --model_path ${output_path}env/user_${MODEL}_lr${LR}_reg${REG}.model\
            --user_latent_dim 16\
            --item_latent_dim 16\
            --enc_dim 32\
            --n_ensemble 2\
            --attn_n_head 4\
            --transformer_d_forward 64\
            --transformer_n_layer 2\
            --state_hidden_dims 128\
            --scorer_hidden_dims 128 32\
            --dropout_rate 0.1
#             > ${output_path}env/log/user_${MODEL}_lr${LR}_reg${REG}.model.log
    done
done