#!/bin/bash
python -u run_fine_tuning_bert4keras_models.py --config_path /data0/nfs_data/zhaoxi9/pretrained_language_model/albert/albert_base/albert_config.json --checkpoint_path /data0/nfs_data/zhaoxi9/pretrained_language_model/albert/albert_base/model.ckpt-best --vocab_path /data0/nfs_data/zhaoxi9/pretrained_language_model/albert/albert_base/vocab_chinese.txt --model_type albert --gpu 3 --batch_size 256 --freeze &&
#Epoch 23/10000
#188/188 - 70s - loss: 1.6690 - accuracy: 0.4641
#dev acc=48.56%
#test acc=49.02%
python -u run_fine_tuning_bert4keras_models.py --config_path /data0/nfs_data/zhaoxi9/pretrained_language_model/albert/albert_large/albert_config.json --checkpoint_path /data0/nfs_data/zhaoxi9/pretrained_language_model/albert/albert_large/model.ckpt-best --vocab_path /data0/nfs_data/zhaoxi9/pretrained_language_model/albert/albert_large/vocab_chinese.txt --model_type albert --gpu 3 --batch_size 256 --freeze &&
#Epoch 16/10000
#188/188 - 218s - loss: 1.5591 - accuracy: 0.4963
#dev acc=50.22%
#test acc=50.79%
python -u run_fine_tuning_bert4keras_models.py --config_path /data0/nfs_data/zhaoxi9/pretrained_language_model/albert/albert_xlarge/albert_config.json --checkpoint_path /data0/nfs_data/zhaoxi9/pretrained_language_model/albert/albert_xlarge/model.ckpt-best --vocab_path /data0/nfs_data/zhaoxi9/pretrained_language_model/albert/albert_xlarge/vocab_chinese.txt --model_type albert --gpu 3 --batch_size 256 --freeze &&
#Epoch 9/10000
#188/188 - 715s - loss: 1.5389 - accuracy: 0.4979
#dev acc=50.82%
#test acc=50.77%
python -u run_fine_tuning_bert4keras_models.py --config_path /data0/nfs_data/zhaoxi9/pretrained_language_model/albert/albert_xxlarge/albert_config.json --checkpoint_path /data0/nfs_data/zhaoxi9/pretrained_language_model/albert/albert_xxlarge/model.ckpt-best --vocab_path /data0/nfs_data/zhaoxi9/pretrained_language_model/albert/albert_xxlarge/vocab_chinese.txt --model_type albert --gpu 3 --batch_size 256 --freeze &&
#Epoch 8/10000
#188/188 - 1348s - loss: 1.5353 - accuracy: 0.5046
#dev acc=50.22%
#test acc=50.08%
python -u run_fine_tuning_bert4keras_models.py --config_path /data0/nfs_data/zhaoxi9/pretrained_language_model/albert_zh/albert_base_google_zh_additional_36k_steps/albert_config.json --checkpoint_path /data0/nfs_data/zhaoxi9/pretrained_language_model/albert_zh/albert_base_google_zh_additional_36k_steps/albert_model.ckpt --vocab_path /data0/nfs_data/zhaoxi9/pretrained_language_model/albert_zh/albert_base_google_zh_additional_36k_steps/vocab.txt --model_type albert --gpu 3 --batch_size 256 --freeze &&
#Epoch 13/10000
#188/188 - 91s - loss: 1.5681 - accuracy: 0.4844
#dev acc=50.45%
#test acc=50.35%
# python -u run_fine_tuning_bert4keras_models.py --config_path /data0/nfs_data/zhaoxi9/pretrained_language_model/albert_zh/albert_base_zh/albert_config_base.json --checkpoint_path /data0/nfs_data/zhaoxi9/pretrained_language_model/albert_zh/albert_base_zh/albert_model.ckpt --vocab_path /data0/nfs_data/zhaoxi9/pretrained_language_model/albert_zh/albert_base_zh/vocab.txt --model_type albert --gpu 3 --batch_size 256 --freeze &&
#RuntimeError: Key bert/encoder/embedding_hidden_mapping_in/kernel not found in checkpoint
# python -u run_fine_tuning_bert4keras_models.py --config_path /data0/nfs_data/zhaoxi9/pretrained_language_model/albert_zh/albert_base_zh_additional_36k_steps/albert_config_base.json --checkpoint_path /data0/nfs_data/zhaoxi9/pretrained_language_model/albert_zh/albert_base_zh_additional_36k_steps/albert_model.ckpt --vocab_path /data0/nfs_data/zhaoxi9/pretrained_language_model/albert_zh/albert_base_zh_additional_36k_steps/vocab.txt --model_type albert --gpu 3 --batch_size 256 --freeze &&
#tensorflow.python.framework.errors_impl.NotFoundError: Key bert/encoder/embedding_hidden_mapping_in/kernel not found in checkpoint
python -u run_fine_tuning_bert4keras_models.py --config_path /data0/nfs_data/zhaoxi9/pretrained_language_model/albert_zh/albert_large_google_zh/albert_config.json --checkpoint_path /data0/nfs_data/zhaoxi9/pretrained_language_model/albert_zh/albert_large_google_zh/albert_model.ckpt --vocab_path /data0/nfs_data/zhaoxi9/pretrained_language_model/albert_zh/albert_large_google_zh/vocab.txt --model_type albert --gpu 3 --batch_size 256 --freeze &&
#Epoch 17/10000
#188/188 - 273s - loss: 1.5237 - accuracy: 0.5054
#dev acc=51.91%
#test acc=51.88%
# python -u run_fine_tuning_bert4keras_models.py --config_path /data0/nfs_data/zhaoxi9/pretrained_language_model/albert_zh/albert_large_zh/albert_config_large.json --checkpoint_path /data0/nfs_data/zhaoxi9/pretrained_language_model/albert_zh/albert_large_zh/albert_model.ckpt --vocab_path /data0/nfs_data/zhaoxi9/pretrained_language_model/albert_zh/albert_large_zh/vocab.txt --model_type albert --gpu 3 --batch_size 256 --freeze &&
#tensorflow.python.framework.errors_impl.NotFoundError: Key bert/encoder/embedding_hidden_mapping_in/kernel not found in checkpoint
python -u run_fine_tuning_bert4keras_models.py --config_path /data0/nfs_data/zhaoxi9/pretrained_language_model/albert_zh/albert_small_zh_google/albert_config_small_google.json --checkpoint_path /data0/nfs_data/zhaoxi9/pretrained_language_model/albert_zh/albert_small_zh_google/albert_model.ckpt --vocab_path /data0/nfs_data/zhaoxi9/pretrained_language_model/albert_zh/albert_small_zh_google/vocab.txt --model_type albert --gpu 3 --batch_size 256 --freeze &&
#Epoch 26/10000
#188/188 - 20s - loss: 1.6925 - accuracy: 0.4492
#dev acc=47.64%
#test acc=48.42%
# python -u run_fine_tuning_bert4keras_models.py --config_path /data0/nfs_data/zhaoxi9/pretrained_language_model/albert_zh/albert_tiny/albert_config_tiny.json --checkpoint_path /data0/nfs_data/zhaoxi9/pretrained_language_model/albert_zh/albert_tiny/albert_model.ckpt --vocab_path /data0/nfs_data/zhaoxi9/pretrained_language_model/albert_zh/albert_tiny/vocab.txt --model_type albert --gpu 3 --batch_size 256 --freeze &&
#tensorflow.python.framework.errors_impl.NotFoundError: Key bert/encoder/embedding_hidden_mapping_in/kernel not found in checkpoint
# python -u run_fine_tuning_bert4keras_models.py --config_path /data0/nfs_data/zhaoxi9/pretrained_language_model/albert_zh/albert_tiny_489k/albert_config_tiny.json --checkpoint_path /data0/nfs_data/zhaoxi9/pretrained_language_model/albert_zh/albert_tiny_489k/albert_model.ckpt --vocab_path /data0/nfs_data/zhaoxi9/pretrained_language_model/albert_zh/albert_tiny_489k/vocab.txt --model_type albert --gpu 3 --batch_size 256 --freeze &&
#tensorflow.python.framework.errors_impl.NotFoundError: Key bert/encoder/embedding_hidden_mapping_in/kernel not found in checkpoint
python -u run_fine_tuning_bert4keras_models.py --config_path /data0/nfs_data/zhaoxi9/pretrained_language_model/albert_zh/albert_tiny_google_zh_489k/albert_config.json --checkpoint_path /data0/nfs_data/zhaoxi9/pretrained_language_model/albert_zh/albert_tiny_google_zh_489k/albert_model.ckpt --vocab_path /data0/nfs_data/zhaoxi9/pretrained_language_model/albert_zh/albert_tiny_google_zh_489k/vocab.txt --model_type albert --gpu 3 --batch_size 256 --freeze &&
#Epoch 16/10000
#188/188 - 16s - loss: 1.7594 - accuracy: 0.4339
#dev acc=46.12%
#test acc=46.70%
python -u run_fine_tuning_bert4keras_models.py --config_path /data0/nfs_data/zhaoxi9/pretrained_language_model/albert_zh/albert_tiny_zh_google/albert_config_tiny_g.json --checkpoint_path /data0/nfs_data/zhaoxi9/pretrained_language_model/albert_zh/albert_tiny_zh_google/albert_model.ckpt --vocab_path /data0/nfs_data/zhaoxi9/pretrained_language_model/albert_zh/albert_tiny_zh_google/vocab.txt --model_type albert --gpu 3 --batch_size 256 --freeze &&
#Epoch 18/10000
#188/188 - 15s - loss: 1.7625 - accuracy: 0.4287
#dev acc=46.74%
#test acc=46.95%
python -u run_fine_tuning_bert4keras_models.py --config_path /data0/nfs_data/zhaoxi9/pretrained_language_model/albert_zh/albert_xlarge_google_zh_183k/albert_config.json --checkpoint_path /data0/nfs_data/zhaoxi9/pretrained_language_model/albert_zh/albert_xlarge_google_zh_183k/albert_model.ckpt --vocab_path /data0/nfs_data/zhaoxi9/pretrained_language_model/albert_zh/albert_xlarge_google_zh_183k/vocab.txt --model_type albert --gpu 3 --batch_size 256 --freeze &&
#Epoch 13/10000
#188/188 - 828s - loss: 1.4085 - accuracy: 0.5338
#dev acc=53.13%
#test acc=52.87%
#python -u run_fine_tuning_bert4keras_models.py --config_path /data0/nfs_data/zhaoxi9/pretrained_language_model/albert_zh/albert_xlarge_zh_177k/albert_config_xlarge.json --checkpoint_path /data0/nfs_data/zhaoxi9/pretrained_language_model/albert_zh/albert_xlarge_zh_177k/albert_model.ckpt --vocab_path /data0/nfs_data/zhaoxi9/pretrained_language_model/albert_zh/albert_xlarge_zh_177k/vocab.txt --model_type albert --gpu 3 --batch_size 256 --freeze &&
# error
#python -u run_fine_tuning_bert4keras_models.py --config_path /data0/nfs_data/zhaoxi9/pretrained_language_model/albert_zh/albert_xlarge_zh_183k/albert_config_xlarge.json --checkpoint_path /data0/nfs_data/zhaoxi9/pretrained_language_model/albert_zh/albert_xlarge_zh_183k/albert_model.ckpt --vocab_path /data0/nfs_data/zhaoxi9/pretrained_language_model/albert_zh/albert_xlarge_zh_183k/vocab.txt --model_type albert --gpu 3 --batch_size 256 --freeze &&
# error

python -u run_fine_tuning_bert4keras_models.py --config_path /data0/nfs_data/zhaoxi9/pretrained_language_model/bert/chinese_L-12_H-768_A-12/bert_config.json --checkpoint_path /data0/nfs_data/zhaoxi9/pretrained_language_model/bert/chinese_L-12_H-768_A-12/bert_model.ckpt --vocab_path /data0/nfs_data/zhaoxi9/pretrained_language_model/bert/chinese_L-12_H-768_A-12/vocab.txt --model_type bert --gpu 3 --batch_size 256 --freeze &&
#Epoch 10/10000
#188/188 - 98s - loss: 1.3520 - accuracy: 0.5408
#dev acc=55.06%
#test acc=55.26%

#python -u run_fine_tuning_bert4keras_models.py --config_path /data0/nfs_data/zhaoxi9/pretrained_language_model/Chinese-BERT-wwm/chinese-bert_chinese_wwm_L-12_H-768_A-12/publish/bert_config.json --checkpoint_path /data0/nfs_data/zhaoxi9/pretrained_language_model/Chinese-BERT-wwm/chinese-bert_chinese_wwm_L-12_H-768_A-12/publish/bert_model.ckpt --vocab_path /data0/nfs_data/zhaoxi9/pretrained_language_model/Chinese-BERT-wwm/chinese-bert_chinese_wwm_L-12_H-768_A-12/publish/vocab.txt --model_type albert --gpu 3 --batch_size 256 --freeze &&
#tensorflow.python.framework.errors_impl.NotFoundError: Key bert/encoder/transformer/group_0/inner_group_0/attention_1/self/query/kernel not found in checkpoint, 奇怪
python -u run_fine_tuning_bert4keras_models.py --config_path /data0/nfs_data/zhaoxi9/pretrained_language_model/Chinese-BERT-wwm/chinese_rbt3_L-3_H-768_A-12/bert_config_rbt3.json --checkpoint_path /data0/nfs_data/zhaoxi9/pretrained_language_model/Chinese-BERT-wwm/chinese_rbt3_L-3_H-768_A-12/bert_model.ckpt --vocab_path /data0/nfs_data/zhaoxi9/pretrained_language_model/Chinese-BERT-wwm/chinese_rbt3_L-3_H-768_A-12/vocab.txt --model_type bert --gpu 3 --batch_size 256 --freeze &&
#Epoch 9/10000
#188/188 - 25s - loss: 1.4725 - accuracy: 0.5115
#dev acc=52.25%
#test acc=53.11%
python -u run_fine_tuning_bert4keras_models.py --config_path /data0/nfs_data/zhaoxi9/pretrained_language_model/Chinese-BERT-wwm/chinese_rbtl3_L-3_H-1024_A-16/bert_config_rbtl3.json --checkpoint_path /data0/nfs_data/zhaoxi9/pretrained_language_model/Chinese-BERT-wwm/chinese_rbtl3_L-3_H-1024_A-16/bert_model.ckpt --vocab_path /data0/nfs_data/zhaoxi9/pretrained_language_model/Chinese-BERT-wwm/chinese_rbtl3_L-3_H-1024_A-16/vocab.txt --model_type bert --gpu 3 --batch_size 256 --freeze &&
#Epoch 24/10000
#188/188 - 38s - loss: 1.4385 - accuracy: 0.5215
#dev acc=53.99%
#test acc=53.43%
python -u run_fine_tuning_bert4keras_models.py --config_path /data0/nfs_data/zhaoxi9/pretrained_language_model/Chinese-BERT-wwm/chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_config.json --checkpoint_path /data0/nfs_data/zhaoxi9/pretrained_language_model/Chinese-BERT-wwm/chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_model.ckpt --vocab_path /data0/nfs_data/zhaoxi9/pretrained_language_model/Chinese-BERT-wwm/chinese_roberta_wwm_ext_L-12_H-768_A-12/vocab.txt --model_type bert --gpu 3 --batch_size 256 --freeze &&
#Epoch 5/10000
#188/188 - 97s - loss: 1.3256 - accuracy: 0.5510
#dev acc=56.63%
#test acc=56.22%
python -u run_fine_tuning_bert4keras_models.py --config_path /data0/nfs_data/zhaoxi9/pretrained_language_model/Chinese-BERT-wwm/chinese_roberta_wwm_large_ext_L-24_H-1024_A-16/bert_config.json --checkpoint_path /data0/nfs_data/zhaoxi9/pretrained_language_model/Chinese-BERT-wwm/chinese_roberta_wwm_large_ext_L-24_H-1024_A-16/bert_model.ckpt --vocab_path /data0/nfs_data/zhaoxi9/pretrained_language_model/Chinese-BERT-wwm/chinese_roberta_wwm_large_ext_L-24_H-1024_A-16/vocab.txt --model_type bert --gpu 3 --batch_size 256 --freeze &&
#Epoch 10/10000
#188/188 - 294s - loss: 1.3002 - accuracy: 0.5567
#dev acc=56.90%
#test acc=56.60%
python -u run_fine_tuning_bert4keras_models.py --config_path /data0/nfs_data/zhaoxi9/pretrained_language_model/Chinese-BERT-wwm/chinese_wwm_ext_L-12_H-768_A-12/bert_config.json --checkpoint_path /data0/nfs_data/zhaoxi9/pretrained_language_model/Chinese-BERT-wwm/chinese_wwm_ext_L-12_H-768_A-12/bert_model.ckpt --vocab_path /data0/nfs_data/zhaoxi9/pretrained_language_model/Chinese-BERT-wwm/chinese_wwm_ext_L-12_H-768_A-12/vocab.txt --model_type bert --gpu 3 --batch_size 256 --freeze &&
#Epoch 15/10000
#188/188 - 99s - loss: 1.3384 - accuracy: 0.5484
#dev acc=56.15%
#test acc=55.96%
python -u run_fine_tuning_bert4keras_models.py --config_path /data0/nfs_data/zhaoxi9/pretrained_language_model/Chinese-ELECTRA/chinese_electra_base_L-12_H-768_A-12/config.json --checkpoint_path /data0/nfs_data/zhaoxi9/pretrained_language_model/Chinese-ELECTRA/chinese_electra_base_L-12_H-768_A-12/electra_base --vocab_path /data0/nfs_data/zhaoxi9/pretrained_language_model/Chinese-ELECTRA/chinese_electra_base_L-12_H-768_A-12/vocab.txt --model_type electra --gpu 3 --batch_size 256 --freeze &&
#Epoch 28/10000
#188/188 - 99s - loss: 2.2427 - accuracy: 0.2685
#dev acc=30.51%
#test acc=30.35%
python -u run_fine_tuning_bert4keras_models.py --config_path /data0/nfs_data/zhaoxi9/pretrained_language_model/Chinese-ELECTRA/chinese_electra_small_L-12_H-256_A-4/config.json --checkpoint_path /data0/nfs_data/zhaoxi9/pretrained_language_model/Chinese-ELECTRA/chinese_electra_small_L-12_H-256_A-4/electra_small --vocab_path /data0/nfs_data/zhaoxi9/pretrained_language_model/Chinese-ELECTRA/chinese_electra_small_L-12_H-256_A-4/vocab.txt --model_type electra --gpu 3 --batch_size 256 --freeze &&
#Epoch 40/10000
#188/188 - 27s - loss: 2.3043 - accuracy: 0.2494
#dev acc=30.42%
#test acc=30.32%

# python -u run_fine_tuning_bert4keras_models.py --config_path /data0/nfs_data/zhaoxi9/pretrained_language_model/Chinese-XLNet/chinese_xlnet_base_L-12_H-768_A-12/xlnet_config.json --checkpoint_path /data0/nfs_data/zhaoxi9/pretrained_language_model/Chinese-XLNet/chinese_xlnet_base_L-12_H-768_A-12/xlnet_model.ckpt --vocab_path /data0/nfs_data/zhaoxi9/pretrained_language_model/Chinese-XLNet/chinese_xlnet_base_L-12_H-768_A-12/spiece.model
# python -u run_fine_tuning_bert4keras_models.py --config_path /data0/nfs_data/zhaoxi9/pretrained_language_model/Chinese-XLNet/chinese_xlnet_mid_L-24_H-768_A-12/xlnet_config.json --checkpoint_path /data0/nfs_data/zhaoxi9/pretrained_language_model/Chinese-XLNet/chinese_xlnet_mid_L-24_H-768_A-12/xlnet_model.ckpt --vocab_path /data0/nfs_data/zhaoxi9/pretrained_language_model/Chinese-XLNet/chinese_xlnet_mid_L-24_H-768_A-12/spiece.model

python -u run_fine_tuning_bert4keras_models.py --config_path /data0/nfs_data/zhaoxi9/pretrained_language_model/ELECTRA/electra_tiny/bert_config_tiny.json --checkpoint_path /data0/nfs_data/zhaoxi9/pretrained_language_model/ELECTRA/electra_tiny/model.ckpt-1000000 --vocab_path /data0/nfs_data/zhaoxi9/pretrained_language_model/ELECTRA/electra_tiny/vocab.txt --model_type electra --gpu 3 --batch_size 256 --freeze &&
#Epoch 21/10000
#188/188 - 13s - loss: 2.2243 - accuracy: 0.2791
#dev acc=32.81%
#test acc=32.52%
python -u run_fine_tuning_bert4keras_models.py --config_path /data0/nfs_data/zhaoxi9/pretrained_language_model/roberta_zh/roberta_zh_l12/bert_config.json --checkpoint_path /data0/nfs_data/zhaoxi9/pretrained_language_model/roberta_zh/roberta_zh_l12/bert_model.ckpt --vocab_path /data0/nfs_data/zhaoxi9/pretrained_language_model/roberta_zh/roberta_zh_l12/vocab.txt --model_type bert --gpu 3 --batch_size 256 --freeze &&
#Epoch 11/10000
#188/188 - 98s - loss: 1.3391 - accuracy: 0.5490
#dev acc=55.94%
#test acc=55.89%
python -u run_fine_tuning_bert4keras_models.py --config_path /data0/nfs_data/zhaoxi9/pretrained_language_model/roberta_zh/roberta_zh_L-6-H-768_A-12/bert_config.json --checkpoint_path /data0/nfs_data/zhaoxi9/pretrained_language_model/roberta_zh/roberta_zh_L-6-H-768_A-12/bert_model.ckpt --vocab_path /data0/nfs_data/zhaoxi9/pretrained_language_model/roberta_zh/roberta_zh_L-6-H-768_A-12/vocab.txt --model_type bert --gpu 3 --batch_size 256 --freeze &&
#Epoch 15/10000
#188/188 - 50s - loss: 1.4454 - accuracy: 0.5187
#dev acc=54.63%
#test acc=54.19%
python -u run_fine_tuning_bert4keras_models.py --config_path /data0/nfs_data/zhaoxi9/pretrained_language_model/roberta_zh/roeberta_zh_L-24_H-1024_A-16/bert_config_large.json --checkpoint_path /data0/nfs_data/zhaoxi9/pretrained_language_model/roberta_zh/roeberta_zh_L-24_H-1024_A-16/roberta_zh_large_model.ckpt --vocab_path /data0/nfs_data/zhaoxi9/pretrained_language_model/roberta_zh/roeberta_zh_L-24_H-1024_A-16/vocab.txt --model_type bert --gpu 3 --batch_size 256 --freeze &&
#Epoch 12/10000
#188/188 - 293s - loss: 1.3471 - accuracy: 0.5464
#dev acc=55.92%
#test acc=55.88%

# python -u run_fine_tuning_bert4keras_models.py --config_path /data0/nfs_data/zhaoxi9/pretrained_language_model/xlnet_zh/XLNet_zh_Large_L-24_H-1024_A-16/xlnet_config.json --checkpoint_path /data0/nfs_data/zhaoxi9/pretrained_language_model/xlnet_zh/XLNet_zh_Large_L-24_H-1024_A-16/xlnet_model --vocab_path /data0/nfs_data/zhaoxi9/pretrained_language_model/xlnet_zh/XLNet_zh_Large_L-24_H-1024_A-16/spiece.model

python -u run_fine_tuning_bert4keras_models.py --config_path /data0/nfs_data/zhaoxi9/pretrained_language_model/zhuiyi/chinese_roberta_L-4_H-312_A-12/bert_config.json --checkpoint_path /data0/nfs_data/zhaoxi9/pretrained_language_model/zhuiyi/chinese_roberta_L-4_H-312_A-12/bert_model.ckpt --vocab_path /data0/nfs_data/zhaoxi9/pretrained_language_model/zhuiyi/chinese_roberta_L-4_H-312_A-12/vocab.txt --model_type bert --gpu 3 --batch_size 256 --freeze &&
#Epoch 24/10000
#188/188 - 16s - loss: 1.7752 - accuracy: 0.4254
#dev acc=46.98%
#test acc=47.67%
python -u run_fine_tuning_bert4keras_models.py --config_path /data0/nfs_data/zhaoxi9/pretrained_language_model/zhuiyi/chinese_roberta_L-4_H-312_A-12_K-104/bert_config.json --checkpoint_path /data0/nfs_data/zhaoxi9/pretrained_language_model/zhuiyi/chinese_roberta_L-4_H-312_A-12_K-104/bert_model.ckpt --vocab_path /data0/nfs_data/zhaoxi9/pretrained_language_model/zhuiyi/chinese_roberta_L-4_H-312_A-12_K-104/vocab.txt --model_type bert --gpu 3 --batch_size 256 --freeze &&
#Epoch 20/10000
#188/188 - 16s - loss: 1.7569 - accuracy: 0.4352
#dev acc=47.75%
#test acc=48.04%
python -u run_fine_tuning_bert4keras_models.py --config_path /data0/nfs_data/zhaoxi9/pretrained_language_model/zhuiyi/chinese_roberta_L-6_H-384_A-12/bert_config.json --checkpoint_path /data0/nfs_data/zhaoxi9/pretrained_language_model/zhuiyi/chinese_roberta_L-6_H-384_A-12/bert_model.ckpt --vocab_path /data0/nfs_data/zhaoxi9/pretrained_language_model/zhuiyi/chinese_roberta_L-6_H-384_A-12/vocab.txt --model_type bert --gpu 3 --batch_size 256 --freeze &&
#Epoch 7/10000
#188/188 - 20s - loss: 1.7317 - accuracy: 0.4394
#dev acc=47.34%
#test acc=48.37%
python -u run_fine_tuning_bert4keras_models.py --config_path /data0/nfs_data/zhaoxi9/pretrained_language_model/zhuiyi/chinese_roberta_L-6_H-384_A-12_K-128/bert_config.json --checkpoint_path /data0/nfs_data/zhaoxi9/pretrained_language_model/zhuiyi/chinese_roberta_L-6_H-384_A-12_K-128/bert_model.ckpt --vocab_path /data0/nfs_data/zhaoxi9/pretrained_language_model/zhuiyi/chinese_roberta_L-6_H-384_A-12_K-128/vocab.txt --model_type bert --gpu 3 --batch_size 256 --freeze &&
#Epoch 18/10000
#188/188 - 26s - loss: 1.7130 - accuracy: 0.4440
#dev acc=48.29%
#test acc=48.40%
python -u run_fine_tuning_bert4keras_models.py --config_path /data0/nfs_data/zhaoxi9/pretrained_language_model/zhuiyi/chinese_simbert_L-12_H-768_A-12/bert_config.json --checkpoint_path /data0/nfs_data/zhaoxi9/pretrained_language_model/zhuiyi/chinese_simbert_L-12_H-768_A-12/bert_model.ckpt --vocab_path /data0/nfs_data/zhaoxi9/pretrained_language_model/zhuiyi/chinese_simbert_L-12_H-768_A-12/vocab.txt --model_type bert --gpu 3 --batch_size 256 --freeze &&
#Epoch 10/10000
#188/188 - 98s - loss: 1.3462 - accuracy: 0.5433
#dev acc=56.07%
#test acc=55.65%
python -u run_fine_tuning_bert4keras_models.py --config_path /data0/nfs_data/zhaoxi9/pretrained_language_model/zhuiyi/chinese_simbert_L-4_H-312_A-12/bert_config.json --checkpoint_path /data0/nfs_data/zhaoxi9/pretrained_language_model/zhuiyi/chinese_simbert_L-4_H-312_A-12/bert_model.ckpt --vocab_path /data0/nfs_data/zhaoxi9/pretrained_language_model/zhuiyi/chinese_simbert_L-4_H-312_A-12/vocab.txt --model_type bert --gpu 3 --batch_size 256 --freeze &&
#Epoch 8/10000
#188/188 - 16s - loss: 1.7777 - accuracy: 0.4266
#dev acc=47.26%
#test acc=47.81%
python -u run_fine_tuning_bert4keras_models.py --config_path /data0/nfs_data/zhaoxi9/pretrained_language_model/zhuiyi/chinese_simbert_L-6_H-384_A-12/bert_config.json --checkpoint_path /data0/nfs_data/zhaoxi9/pretrained_language_model/zhuiyi/chinese_simbert_L-6_H-384_A-12/bert_model.ckpt --vocab_path /data0/nfs_data/zhaoxi9/pretrained_language_model/zhuiyi/chinese_simbert_L-6_H-384_A-12/vocab.txt --model_type bert --gpu 3 --batch_size 256 --freeze &&
#Epoch 21/10000
#188/188 - 20s - loss: 1.7170 - accuracy: 0.4464
#dev acc=48.63%
#test acc=49.26%
python -u run_fine_tuning_bert4keras_models.py --config_path /data0/nfs_data/zhaoxi9/pretrained_language_model/zhuiyi/chinese_wobert_L-12_H-768_A-12/bert_config.json --checkpoint_path /data0/nfs_data/zhaoxi9/pretrained_language_model/zhuiyi/chinese_wobert_L-12_H-768_A-12/bert_model.ckpt --vocab_path /data0/nfs_data/zhaoxi9/pretrained_language_model/zhuiyi/chinese_wobert_L-12_H-768_A-12/vocab.txt --model_type bert --gpu 3 --batch_size 256 --freeze &&
#Epoch 7/10000
#188/188 - 98s - loss: 1.3362 - accuracy: 0.5475
#dev acc=56.09%
#test acc=56.21%
python -u run_fine_tuning_bert4keras_models.py --config_path /data0/nfs_data/zhaoxi9/pretrained_language_model/zhuiyi/chinese_wobert_plus_L-12_H-768_A-12/bert_config.json --checkpoint_path /data0/nfs_data/zhaoxi9/pretrained_language_model/zhuiyi/chinese_wobert_plus_L-12_H-768_A-12/bert_model.ckpt --vocab_path /data0/nfs_data/zhaoxi9/pretrained_language_model/zhuiyi/chinese_wobert_plus_L-12_H-768_A-12/vocab.txt --model_type bert --gpu 3 --batch_size 256 --freeze &&
#Epoch 7/10000
#188/188 - 98s - loss: 1.3561 - accuracy: 0.5424
#dev acc=55.66%
#test acc=55.68%
python -u run_fine_tuning_bert4keras_models.py --config_path /data0/nfs_data/zhaoxi9/pretrained_language_model/zhuiyi/chinese_wonezha_L-12_H-768_A-12/bert_config.json --checkpoint_path /data0/nfs_data/zhaoxi9/pretrained_language_model/zhuiyi/chinese_wonezha_L-12_H-768_A-12/bert_model.ckpt --vocab_path /data0/nfs_data/zhaoxi9/pretrained_language_model/zhuiyi/chinese_wonezha_L-12_H-768_A-12/vocab.txt --model_type nezha --gpu 3 --batch_size 256 --freeze
#Epoch 17/10000
#188/188 - 104s - loss: 1.4068 - accuracy: 0.5327
#dev acc=55.85%
#test acc=56.07%