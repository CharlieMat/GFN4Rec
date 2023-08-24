mkdir -p output

mkdir -p output/kuairand_1k/
mkdir -p output/kuairand_1k/env
mkdir -p output/kuairand_1k/env/log

# data_path='../dataset/kuairand/KuaiRand-Pure/data/'
# train_path=${data_path}log_session_4_08_to_5_08_pure.csv
# user_meta_file=${data_path}user_features_pure_fillna.csv
# item_meta_file=${data_path}video_features_basic_pure_fillna.csv
data_path='../dataset/kuairand/KuaiRand-1K/data/'
train_path=${data_path}log_session_4_08_to_5_08_1k.csv
user_meta_path=${data_path}user_features_1k_fillna.csv
item_meta_path=${data_path}video_features_basic_1k_fillna.csv
output_path='output/kuairand_1k/'

# MODEL='KRMBUserResponse'
MODEL='KRMBUserResponse_MaxOut'

for LR in 0.0001 # 0.00001 0.001
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
            --reader KRMBSeqReader\
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
            --dropout_rate 0.1\
            > ${output_path}env/log/user_${MODEL}_lr${LR}_reg${REG}.model.log
    done
done