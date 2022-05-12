#!/bin/bash
#python -u run_fine_tuning_bert4keras_models.py --config_path /data0/nfs_data/zhaoxi9/pretrained_language_model/albert/albert_base/albert_config.json --checkpoint_path /data0/nfs_data/zhaoxi9/pretrained_language_model/albert/albert_base/model.ckpt-best --vocab_path /data0/nfs_data/zhaoxi9/pretrained_language_model/albert/albert_base/vocab_chinese.txt --model_type albert --gpu 1 --batch_size 64 &&
#Epoch 1/10000
#751/751 - 219s - loss: 2.6153 - accuracy: 0.1046
#dev acc=11.15%
#test acc=10.89%
#python -u run_fine_tuning_bert4keras_models.py --config_path /data0/nfs_data/zhaoxi9/pretrained_language_model/albert/albert_large/albert_config.json --checkpoint_path /data0/nfs_data/zhaoxi9/pretrained_language_model/albert/albert_large/model.ckpt-best --vocab_path /data0/nfs_data/zhaoxi9/pretrained_language_model/albert/albert_large/vocab_chinese.txt --model_type albert --gpu 1 --batch_size 64 &&
#Epoch 1/10000
#751/751 - 635s - loss: 2.6244 - accuracy: 0.1014
#dev acc=11.15%
#test acc=10.89%
#python -u run_fine_tuning_bert4keras_models.py --config_path /data0/nfs_data/zhaoxi9/pretrained_language_model/albert/albert_xlarge/albert_config.json --checkpoint_path /data0/nfs_data/zhaoxi9/pretrained_language_model/albert/albert_xlarge/model.ckpt-best --vocab_path /data0/nfs_data/zhaoxi9/pretrained_language_model/albert/albert_xlarge/vocab_chinese.txt --model_type albert --gpu 1 --batch_size 64 &&
#Epoch 4/10000
#751/751 - 1937s - loss: 2.6140 - accuracy: 0.1047
#dev acc=11.15%
#test acc=10.89%
#python -u run_fine_tuning_bert4keras_models.py --config_path /data0/nfs_data/zhaoxi9/pretrained_language_model/albert/albert_xxlarge/albert_config.json --checkpoint_path /data0/nfs_data/zhaoxi9/pretrained_language_model/albert/albert_xxlarge/model.ckpt-best --vocab_path /data0/nfs_data/zhaoxi9/pretrained_language_model/albert/albert_xxlarge/vocab_chinese.txt --model_type albert --gpu 1 --batch_size 64 &&
#Epoch 1/10000
#751/751 - 3616s - loss: 2.6902 - accuracy: 0.0940
#dev acc=9.09%
#test acc=9.05%
#python -u run_fine_tuning_bert4keras_models.py --config_path /data0/nfs_data/zhaoxi9/pretrained_language_model/albert_zh/albert_base_google_zh_additional_36k_steps/albert_config.json --checkpoint_path /data0/nfs_data/zhaoxi9/pretrained_language_model/albert_zh/albert_base_google_zh_additional_36k_steps/albert_model.ckpt --vocab_path /data0/nfs_data/zhaoxi9/pretrained_language_model/albert_zh/albert_base_google_zh_additional_36k_steps/vocab.txt --model_type albert --gpu 1 --batch_size 64 &&
#Epoch 3/10000
#751/751 - 256s - loss: 2.6214 - accuracy: 0.1010
#dev acc=11.15%
#test acc=10.89%
# python -u run_fine_tuning_bert4keras_models.py --config_path /data0/nfs_data/zhaoxi9/pretrained_language_model/albert_zh/albert_base_zh/albert_config_base.json --checkpoint_path /data0/nfs_data/zhaoxi9/pretrained_language_model/albert_zh/albert_base_zh/albert_model.ckpt --vocab_path /data0/nfs_data/zhaoxi9/pretrained_language_model/albert_zh/albert_base_zh/vocab.txt --model_type albert --gpu 1 --batch_size 64 &&
#RuntimeError: Key bert/encoder/embedding_hidden_mapping_in/kernel not found in checkpoint
# python -u run_fine_tuning_bert4keras_models.py --config_path /data0/nfs_data/zhaoxi9/pretrained_language_model/albert_zh/albert_base_zh_additional_36k_steps/albert_config_base.json --checkpoint_path /data0/nfs_data/zhaoxi9/pretrained_language_model/albert_zh/albert_base_zh_additional_36k_steps/albert_model.ckpt --vocab_path /data0/nfs_data/zhaoxi9/pretrained_language_model/albert_zh/albert_base_zh_additional_36k_steps/vocab.txt --model_type albert --gpu 1 --batch_size 64 &&
#tensorflow.python.framework.errors_impl.NotFoundError: Key bert/encoder/embedding_hidden_mapping_in/kernel not found in checkpoint
#python -u run_fine_tuning_bert4keras_models.py --config_path /data0/nfs_data/zhaoxi9/pretrained_language_model/albert_zh/albert_large_google_zh/albert_config.json --checkpoint_path /data0/nfs_data/zhaoxi9/pretrained_language_model/albert_zh/albert_large_google_zh/albert_model.ckpt --vocab_path /data0/nfs_data/zhaoxi9/pretrained_language_model/albert_zh/albert_large_google_zh/vocab.txt --model_type albert --gpu 1 --batch_size 64 &&
#Epoch 3/10000
#751/751 - 741s - loss: 2.6152 - accuracy: 0.1042
#dev acc=11.15%
#test acc=10.89%
# python -u run_fine_tuning_bert4keras_models.py --config_path /data0/nfs_data/zhaoxi9/pretrained_language_model/albert_zh/albert_large_zh/albert_config_large.json --checkpoint_path /data0/nfs_data/zhaoxi9/pretrained_language_model/albert_zh/albert_large_zh/albert_model.ckpt --vocab_path /data0/nfs_data/zhaoxi9/pretrained_language_model/albert_zh/albert_large_zh/vocab.txt --model_type albert --gpu 1 --batch_size 64 &&
#tensorflow.python.framework.errors_impl.NotFoundError: Key bert/encoder/embedding_hidden_mapping_in/kernel not found in checkpoint
#python -u run_fine_tuning_bert4keras_models.py --config_path /data0/nfs_data/zhaoxi9/pretrained_language_model/albert_zh/albert_small_zh_google/albert_config_small_google.json --checkpoint_path /data0/nfs_data/zhaoxi9/pretrained_language_model/albert_zh/albert_small_zh_google/albert_model.ckpt --vocab_path /data0/nfs_data/zhaoxi9/pretrained_language_model/albert_zh/albert_small_zh_google/vocab.txt --model_type albert --gpu 1 --batch_size 64 &&
#Epoch 1/10000
#751/751 - 67s - loss: 2.4937 - accuracy: 0.1447
#dev acc=15.16%
#test acc=15.23%
# python -u run_fine_tuning_bert4keras_models.py --config_path /data0/nfs_data/zhaoxi9/pretrained_language_model/albert_zh/albert_tiny/albert_config_tiny.json --checkpoint_path /data0/nfs_data/zhaoxi9/pretrained_language_model/albert_zh/albert_tiny/albert_model.ckpt --vocab_path /data0/nfs_data/zhaoxi9/pretrained_language_model/albert_zh/albert_tiny/vocab.txt --model_type albert --gpu 1 --batch_size 64 &&
#tensorflow.python.framework.errors_impl.NotFoundError: Key bert/encoder/embedding_hidden_mapping_in/kernel not found in checkpoint
# python -u run_fine_tuning_bert4keras_models.py --config_path /data0/nfs_data/zhaoxi9/pretrained_language_model/albert_zh/albert_tiny_489k/albert_config_tiny.json --checkpoint_path /data0/nfs_data/zhaoxi9/pretrained_language_model/albert_zh/albert_tiny_489k/albert_model.ckpt --vocab_path /data0/nfs_data/zhaoxi9/pretrained_language_model/albert_zh/albert_tiny_489k/vocab.txt --model_type albert --gpu 1 --batch_size 64 &&
#tensorflow.python.framework.errors_impl.NotFoundError: Key bert/encoder/embedding_hidden_mapping_in/kernel not found in checkpoint
#python -u run_fine_tuning_bert4keras_models.py --config_path /data0/nfs_data/zhaoxi9/pretrained_language_model/albert_zh/albert_tiny_google_zh_489k/albert_config.json --checkpoint_path /data0/nfs_data/zhaoxi9/pretrained_language_model/albert_zh/albert_tiny_google_zh_489k/albert_model.ckpt --vocab_path /data0/nfs_data/zhaoxi9/pretrained_language_model/albert_zh/albert_tiny_google_zh_489k/vocab.txt --model_type albert --gpu 1 --batch_size 64 &&
#Epoch 9/10000
#751/751 - 37s - loss: 1.0237 - accuracy: 0.6745
#dev acc=50.49%
#test acc=49.26%
# epoch_1
#751/751 - 44s - loss: 1.7956 - accuracy: 0.4284
#dev acc=48.73%
#test acc=47.83%
#python -u run_fine_tuning_bert4keras_models.py --config_path /data0/nfs_data/zhaoxi9/pretrained_language_model/albert_zh/albert_tiny_zh_google/albert_config_tiny_g.json --checkpoint_path /data0/nfs_data/zhaoxi9/pretrained_language_model/albert_zh/albert_tiny_zh_google/albert_model.ckpt --vocab_path /data0/nfs_data/zhaoxi9/pretrained_language_model/albert_zh/albert_tiny_zh_google/vocab.txt --model_type albert --gpu 1 --batch_size 64 &&
#Epoch 11/10000
#751/751 - 35s - loss: 1.5615 - accuracy: 0.5066
#dev acc=48.03%
#test acc=46.72%
# epoch_1
#751/751 - 42s - loss: 1.7865 - accuracy: 0.4354
#dev acc=47.21%
#test acc=47.34%
python -u run_fine_tuning_bert4keras_models.py --config_path /data0/nfs_data/zhaoxi9/pretrained_language_model/albert_zh/albert_xlarge_google_zh_183k/albert_config.json --checkpoint_path /data0/nfs_data/zhaoxi9/pretrained_language_model/albert_zh/albert_xlarge_google_zh_183k/albert_model.ckpt --vocab_path /data0/nfs_data/zhaoxi9/pretrained_language_model/albert_zh/albert_xlarge_google_zh_183k/vocab.txt --model_type albert --gpu 1 --batch_size 64 &&

#python -u run_fine_tuning_bert4keras_models.py --config_path /data0/nfs_data/zhaoxi9/pretrained_language_model/albert_zh/albert_xlarge_zh_177k/albert_config_xlarge.json --checkpoint_path /data0/nfs_data/zhaoxi9/pretrained_language_model/albert_zh/albert_xlarge_zh_177k/albert_model.ckpt --vocab_path /data0/nfs_data/zhaoxi9/pretrained_language_model/albert_zh/albert_xlarge_zh_177k/vocab.txt --model_type albert --gpu 1 --batch_size 64 &&
# error
#python -u run_fine_tuning_bert4keras_models.py --config_path /data0/nfs_data/zhaoxi9/pretrained_language_model/albert_zh/albert_xlarge_zh_183k/albert_config_xlarge.json --checkpoint_path /data0/nfs_data/zhaoxi9/pretrained_language_model/albert_zh/albert_xlarge_zh_183k/albert_model.ckpt --vocab_path /data0/nfs_data/zhaoxi9/pretrained_language_model/albert_zh/albert_xlarge_zh_183k/vocab.txt --model_type albert --gpu 1 --batch_size 64 &&
# error

python -u run_fine_tuning_bert4keras_models.py --config_path /data0/nfs_data/zhaoxi9/pretrained_language_model/bert/chinese_L-12_H-768_A-12/bert_config.json --checkpoint_path /data0/nfs_data/zhaoxi9/pretrained_language_model/bert/chinese_L-12_H-768_A-12/bert_model.ckpt --vocab_path /data0/nfs_data/zhaoxi9/pretrained_language_model/bert/chinese_L-12_H-768_A-12/vocab.txt --model_type bert --gpu 1 --batch_size 64 &&

#python -u run_fine_tuning_bert4keras_models.py --config_path /data0/nfs_data/zhaoxi9/pretrained_language_model/Chinese-BERT-wwm/chinese-bert_chinese_wwm_L-12_H-768_A-12/publish/bert_config.json --checkpoint_path /data0/nfs_data/zhaoxi9/pretrained_language_model/Chinese-BERT-wwm/chinese-bert_chinese_wwm_L-12_H-768_A-12/publish/bert_model.ckpt --vocab_path /data0/nfs_data/zhaoxi9/pretrained_language_model/Chinese-BERT-wwm/chinese-bert_chinese_wwm_L-12_H-768_A-12/publish/vocab.txt --model_type albert --gpu 1 --batch_size 64 &&
#tensorflow.python.framework.errors_impl.NotFoundError: Key bert/encoder/transformer/group_0/inner_group_0/attention_1/self/query/kernel not found in checkpoint, 奇怪
python -u run_fine_tuning_bert4keras_models.py --config_path /data0/nfs_data/zhaoxi9/pretrained_language_model/Chinese-BERT-wwm/chinese_rbt3_L-3_H-768_A-12/bert_config_rbt3.json --checkpoint_path /data0/nfs_data/zhaoxi9/pretrained_language_model/Chinese-BERT-wwm/chinese_rbt3_L-3_H-768_A-12/bert_model.ckpt --vocab_path /data0/nfs_data/zhaoxi9/pretrained_language_model/Chinese-BERT-wwm/chinese_rbt3_L-3_H-768_A-12/vocab.txt --model_type bert --gpu 1 --batch_size 64 &&
# eopch_1
#751/751 - 93s - loss: 1.8814 - accuracy: 0.4131
#dev acc=42.13%
#test acc=41.86%
python -u run_fine_tuning_bert4keras_models.py --config_path /data0/nfs_data/zhaoxi9/pretrained_language_model/Chinese-BERT-wwm/chinese_rbtl3_L-3_H-1024_A-16/bert_config_rbtl3.json --checkpoint_path /data0/nfs_data/zhaoxi9/pretrained_language_model/Chinese-BERT-wwm/chinese_rbtl3_L-3_H-1024_A-16/bert_model.ckpt --vocab_path /data0/nfs_data/zhaoxi9/pretrained_language_model/Chinese-BERT-wwm/chinese_rbtl3_L-3_H-1024_A-16/vocab.txt --model_type bert --gpu 1 --batch_size 64 &&
python -u run_fine_tuning_bert4keras_models.py --config_path /data0/nfs_data/zhaoxi9/pretrained_language_model/Chinese-BERT-wwm/chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_config.json --checkpoint_path /data0/nfs_data/zhaoxi9/pretrained_language_model/Chinese-BERT-wwm/chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_model.ckpt --vocab_path /data0/nfs_data/zhaoxi9/pretrained_language_model/Chinese-BERT-wwm/chinese_roberta_wwm_ext_L-12_H-768_A-12/vocab.txt --model_type bert --gpu 1 --batch_size 64 &&
python -u run_fine_tuning_bert4keras_models.py --config_path /data0/nfs_data/zhaoxi9/pretrained_language_model/Chinese-BERT-wwm/chinese_roberta_wwm_large_ext_L-24_H-1024_A-16/bert_config.json --checkpoint_path /data0/nfs_data/zhaoxi9/pretrained_language_model/Chinese-BERT-wwm/chinese_roberta_wwm_large_ext_L-24_H-1024_A-16/bert_model.ckpt --vocab_path /data0/nfs_data/zhaoxi9/pretrained_language_model/Chinese-BERT-wwm/chinese_roberta_wwm_large_ext_L-24_H-1024_A-16/vocab.txt --model_type bert --gpu 1 --batch_size 64 &&
python -u run_fine_tuning_bert4keras_models.py --config_path /data0/nfs_data/zhaoxi9/pretrained_language_model/Chinese-BERT-wwm/chinese_wwm_ext_L-12_H-768_A-12/bert_config.json --checkpoint_path /data0/nfs_data/zhaoxi9/pretrained_language_model/Chinese-BERT-wwm/chinese_wwm_ext_L-12_H-768_A-12/bert_model.ckpt --vocab_path /data0/nfs_data/zhaoxi9/pretrained_language_model/Chinese-BERT-wwm/chinese_wwm_ext_L-12_H-768_A-12/vocab.txt --model_type bert --gpu 1 --batch_size 64 &&
python -u run_fine_tuning_bert4keras_models.py --config_path /data0/nfs_data/zhaoxi9/pretrained_language_model/Chinese-ELECTRA/chinese_electra_base_L-12_H-768_A-12/config.json --checkpoint_path /data0/nfs_data/zhaoxi9/pretrained_language_model/Chinese-ELECTRA/chinese_electra_base_L-12_H-768_A-12/electra_base --vocab_path /data0/nfs_data/zhaoxi9/pretrained_language_model/Chinese-ELECTRA/chinese_electra_base_L-12_H-768_A-12/vocab.txt --model_type electra --gpu 1 --batch_size 64 &&
python -u run_fine_tuning_bert4keras_models.py --config_path /data0/nfs_data/zhaoxi9/pretrained_language_model/Chinese-ELECTRA/chinese_electra_small_L-12_H-256_A-4/config.json --checkpoint_path /data0/nfs_data/zhaoxi9/pretrained_language_model/Chinese-ELECTRA/chinese_electra_small_L-12_H-256_A-4/electra_small --vocab_path /data0/nfs_data/zhaoxi9/pretrained_language_model/Chinese-ELECTRA/chinese_electra_small_L-12_H-256_A-4/vocab.txt --model_type electra --gpu 1 --batch_size 64 &&

# python -u run_fine_tuning_bert4keras_models.py --config_path /data0/nfs_data/zhaoxi9/pretrained_language_model/Chinese-XLNet/chinese_xlnet_base_L-12_H-768_A-12/xlnet_config.json --checkpoint_path /data0/nfs_data/zhaoxi9/pretrained_language_model/Chinese-XLNet/chinese_xlnet_base_L-12_H-768_A-12/xlnet_model.ckpt --vocab_path /data0/nfs_data/zhaoxi9/pretrained_language_model/Chinese-XLNet/chinese_xlnet_base_L-12_H-768_A-12/spiece.model
# python -u run_fine_tuning_bert4keras_models.py --config_path /data0/nfs_data/zhaoxi9/pretrained_language_model/Chinese-XLNet/chinese_xlnet_mid_L-24_H-768_A-12/xlnet_config.json --checkpoint_path /data0/nfs_data/zhaoxi9/pretrained_language_model/Chinese-XLNet/chinese_xlnet_mid_L-24_H-768_A-12/xlnet_model.ckpt --vocab_path /data0/nfs_data/zhaoxi9/pretrained_language_model/Chinese-XLNet/chinese_xlnet_mid_L-24_H-768_A-12/spiece.model

python -u run_fine_tuning_bert4keras_models.py --config_path /data0/nfs_data/zhaoxi9/pretrained_language_model/ELECTRA/electra_tiny/bert_config_tiny.json --checkpoint_path /data0/nfs_data/zhaoxi9/pretrained_language_model/ELECTRA/electra_tiny/model.ckpt-1000000 --vocab_path /data0/nfs_data/zhaoxi9/pretrained_language_model/ELECTRA/electra_tiny/vocab.txt --model_type electra --gpu 1 --batch_size 64 &&
# epoch_1
#751/751 - 53s - loss: 1.8492 - accuracy: 0.4115
#dev acc=47.19%
#test acc=47.36%
python -u run_fine_tuning_bert4keras_models.py --config_path /data0/nfs_data/zhaoxi9/pretrained_language_model/roberta_zh/roberta_zh_l12/bert_config.json --checkpoint_path /data0/nfs_data/zhaoxi9/pretrained_language_model/roberta_zh/roberta_zh_l12/bert_model.ckpt --vocab_path /data0/nfs_data/zhaoxi9/pretrained_language_model/roberta_zh/roberta_zh_l12/vocab.txt --model_type bert --gpu 1 --batch_size 64 &&
python -u run_fine_tuning_bert4keras_models.py --config_path /data0/nfs_data/zhaoxi9/pretrained_language_model/roberta_zh/roberta_zh_L-6-H-768_A-12/bert_config.json --checkpoint_path /data0/nfs_data/zhaoxi9/pretrained_language_model/roberta_zh/roberta_zh_L-6-H-768_A-12/bert_model.ckpt --vocab_path /data0/nfs_data/zhaoxi9/pretrained_language_model/roberta_zh/roberta_zh_L-6-H-768_A-12/vocab.txt --model_type bert --gpu 1 --batch_size 64 &&
python -u run_fine_tuning_bert4keras_models.py --config_path /data0/nfs_data/zhaoxi9/pretrained_language_model/roberta_zh/roeberta_zh_L-24_H-1024_A-16/bert_config_large.json --checkpoint_path /data0/nfs_data/zhaoxi9/pretrained_language_model/roberta_zh/roeberta_zh_L-24_H-1024_A-16/roberta_zh_large_model.ckpt --vocab_path /data0/nfs_data/zhaoxi9/pretrained_language_model/roberta_zh/roeberta_zh_L-24_H-1024_A-16/vocab.txt --model_type bert --gpu 1 --batch_size 64 &&

# python -u run_fine_tuning_bert4keras_models.py --config_path /data0/nfs_data/zhaoxi9/pretrained_language_model/xlnet_zh/XLNet_zh_Large_L-24_H-1024_A-16/xlnet_config.json --checkpoint_path /data0/nfs_data/zhaoxi9/pretrained_language_model/xlnet_zh/XLNet_zh_Large_L-24_H-1024_A-16/xlnet_model --vocab_path /data0/nfs_data/zhaoxi9/pretrained_language_model/xlnet_zh/XLNet_zh_Large_L-24_H-1024_A-16/spiece.model

python -u run_fine_tuning_bert4keras_models.py --config_path /data0/nfs_data/zhaoxi9/pretrained_language_model/zhuiyi/chinese_roberta_L-4_H-312_A-12/bert_config.json --checkpoint_path /data0/nfs_data/zhaoxi9/pretrained_language_model/zhuiyi/chinese_roberta_L-4_H-312_A-12/bert_model.ckpt --vocab_path /data0/nfs_data/zhaoxi9/pretrained_language_model/zhuiyi/chinese_roberta_L-4_H-312_A-12/vocab.txt --model_type bert --gpu 1 --batch_size 64 &&
# epoch_1
#751/751 - 49s - loss: 1.8551 - accuracy: 0.4146
#dev acc=43.27%
#test acc=44.42%
python -u run_fine_tuning_bert4keras_models.py --config_path /data0/nfs_data/zhaoxi9/pretrained_language_model/zhuiyi/chinese_roberta_L-4_H-312_A-12_K-104/bert_config.json --checkpoint_path /data0/nfs_data/zhaoxi9/pretrained_language_model/zhuiyi/chinese_roberta_L-4_H-312_A-12_K-104/bert_model.ckpt --vocab_path /data0/nfs_data/zhaoxi9/pretrained_language_model/zhuiyi/chinese_roberta_L-4_H-312_A-12_K-104/vocab.txt --model_type bert --gpu 1 --batch_size 64 &&
# epoch_1
#751/751 - 57s - loss: 1.9616 - accuracy: 0.3764
#dev acc=39.81%
#test acc=39.01%
python -u run_fine_tuning_bert4keras_models.py --config_path /data0/nfs_data/zhaoxi9/pretrained_language_model/zhuiyi/chinese_roberta_L-6_H-384_A-12/bert_config.json --checkpoint_path /data0/nfs_data/zhaoxi9/pretrained_language_model/zhuiyi/chinese_roberta_L-6_H-384_A-12/bert_model.ckpt --vocab_path /data0/nfs_data/zhaoxi9/pretrained_language_model/zhuiyi/chinese_roberta_L-6_H-384_A-12/vocab.txt --model_type bert --gpu 1 --batch_size 64 &&
python -u run_fine_tuning_bert4keras_models.py --config_path /data0/nfs_data/zhaoxi9/pretrained_language_model/zhuiyi/chinese_roberta_L-6_H-384_A-12_K-128/bert_config.json --checkpoint_path /data0/nfs_data/zhaoxi9/pretrained_language_model/zhuiyi/chinese_roberta_L-6_H-384_A-12_K-128/bert_model.ckpt --vocab_path /data0/nfs_data/zhaoxi9/pretrained_language_model/zhuiyi/chinese_roberta_L-6_H-384_A-12_K-128/vocab.txt --model_type bert --gpu 1 --batch_size 64 &&
python -u run_fine_tuning_bert4keras_models.py --config_path /data0/nfs_data/zhaoxi9/pretrained_language_model/zhuiyi/chinese_simbert_L-12_H-768_A-12/bert_config.json --checkpoint_path /data0/nfs_data/zhaoxi9/pretrained_language_model/zhuiyi/chinese_simbert_L-12_H-768_A-12/bert_model.ckpt --vocab_path /data0/nfs_data/zhaoxi9/pretrained_language_model/zhuiyi/chinese_simbert_L-12_H-768_A-12/vocab.txt --model_type bert --gpu 1 --batch_size 64 &&
python -u run_fine_tuning_bert4keras_models.py --config_path /data0/nfs_data/zhaoxi9/pretrained_language_model/zhuiyi/chinese_simbert_L-4_H-312_A-12/bert_config.json --checkpoint_path /data0/nfs_data/zhaoxi9/pretrained_language_model/zhuiyi/chinese_simbert_L-4_H-312_A-12/bert_model.ckpt --vocab_path /data0/nfs_data/zhaoxi9/pretrained_language_model/zhuiyi/chinese_simbert_L-4_H-312_A-12/vocab.txt --model_type bert --gpu 1 --batch_size 64 &&
python -u run_fine_tuning_bert4keras_models.py --config_path /data0/nfs_data/zhaoxi9/pretrained_language_model/zhuiyi/chinese_simbert_L-6_H-384_A-12/bert_config.json --checkpoint_path /data0/nfs_data/zhaoxi9/pretrained_language_model/zhuiyi/chinese_simbert_L-6_H-384_A-12/bert_model.ckpt --vocab_path /data0/nfs_data/zhaoxi9/pretrained_language_model/zhuiyi/chinese_simbert_L-6_H-384_A-12/vocab.txt --model_type bert --gpu 1 --batch_size 64 &&
python -u run_fine_tuning_bert4keras_models.py --config_path /data0/nfs_data/zhaoxi9/pretrained_language_model/zhuiyi/chinese_wobert_L-12_H-768_A-12/bert_config.json --checkpoint_path /data0/nfs_data/zhaoxi9/pretrained_language_model/zhuiyi/chinese_wobert_L-12_H-768_A-12/bert_model.ckpt --vocab_path /data0/nfs_data/zhaoxi9/pretrained_language_model/zhuiyi/chinese_wobert_L-12_H-768_A-12/vocab.txt --model_type bert --gpu 1 --batch_size 64 &&
python -u run_fine_tuning_bert4keras_models.py --config_path /data0/nfs_data/zhaoxi9/pretrained_language_model/zhuiyi/chinese_wobert_plus_L-12_H-768_A-12/bert_config.json --checkpoint_path /data0/nfs_data/zhaoxi9/pretrained_language_model/zhuiyi/chinese_wobert_plus_L-12_H-768_A-12/bert_model.ckpt --vocab_path /data0/nfs_data/zhaoxi9/pretrained_language_model/zhuiyi/chinese_wobert_plus_L-12_H-768_A-12/vocab.txt --model_type bert --gpu 1 --batch_size 64 &&
python -u run_fine_tuning_bert4keras_models.py --config_path /data0/nfs_data/zhaoxi9/pretrained_language_model/zhuiyi/chinese_wonezha_L-12_H-768_A-12/bert_config.json --checkpoint_path /data0/nfs_data/zhaoxi9/pretrained_language_model/zhuiyi/chinese_wonezha_L-12_H-768_A-12/bert_model.ckpt --vocab_path /data0/nfs_data/zhaoxi9/pretrained_language_model/zhuiyi/chinese_wonezha_L-12_H-768_A-12/vocab.txt --model_type nezha --gpu 1 --batch_size 64
