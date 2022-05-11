#!/bin/bash
python -u run_fine_tuning_bert4keras_models.py --config_path /data0/nfs_data/zhaoxi9/pretrained_language_model/Chinese-BERT-wwm/chinese_L-12_H-768_A-12/bert_config.json --checkpoint_path /data0/nfs_data/zhaoxi9/pretrained_language_model/Chinese-BERT-wwm/chinese_L-12_H-768_A-12/bert_model.ckpt --vocab_path /data0/nfs_data/zhaoxi9/pretrained_language_model/Chinese-BERT-wwm/chinese_L-12_H-768_A-12/vocab.txt --model_type bert --gpu 2 --batch_size 256 --freeze &&
#Epoch 12/10000
#188/188 - 98s - loss: 1.3505 - accuracy: 0.5417
#dev acc=55.08%
#test acc=55.16%
python -u run_fine_tuning_bert4keras_models.py --config_path /data0/nfs_data/zhaoxi9/pretrained_language_model/Chinese-BERT-wwm/chinese_bert_wwm_L-12_H-768_A-12/publish/bert_config.json --checkpoint_path /data0/nfs_data/zhaoxi9/pretrained_language_model/Chinese-BERT-wwm/chinese_bert_wwm_L-12_H-768_A-12/publish/bert_model.ckpt --vocab_path /data0/nfs_data/zhaoxi9/pretrained_language_model/Chinese-BERT-wwm/chinese_bert_wwm_L-12_H-768_A-12/publish/vocab.txt --model_type bert --gpu 2 --batch_size 256 --freeze &&
#Epoch 14/10000
#188/188 - 98s - loss: 1.3626 - accuracy: 0.5404
#dev acc=55.10%
#test acc=55.34%
python -u run_fine_tuning_bert4keras_models.py --config_path /data0/nfs_data/zhaoxi9/pretrained_language_model/Chinese-BERT-wwm/chinese_rbt4_L-4_H-768_A-12/bert_config_rbt4.json --checkpoint_path /data0/nfs_data/zhaoxi9/pretrained_language_model/Chinese-BERT-wwm/chinese_rbt4_L-4_H-768_A-12/bert_model.ckpt --vocab_path /data0/nfs_data/zhaoxi9/pretrained_language_model/Chinese-BERT-wwm/chinese_rbt4_L-4_H-768_A-12/vocab.txt --model_type bert --gpu 2 --batch_size 256 --freeze &&
#Epoch 15/10000
#188/188 - 34s - loss: 1.4470 - accuracy: 0.5204
#dev acc=53.88%
#test acc=53.61%
python -u run_fine_tuning_bert4keras_models.py --config_path /data0/nfs_data/zhaoxi9/pretrained_language_model/Chinese-BERT-wwm/chinese_rbt6_L-6_H-768_A-12/bert_config_rbt6.json --checkpoint_path /data0/nfs_data/zhaoxi9/pretrained_language_model/Chinese-BERT-wwm/chinese_rbt6_L-6_H-768_A-12/bert_model.ckpt --vocab_path /data0/nfs_data/zhaoxi9/pretrained_language_model/Chinese-BERT-wwm/chinese_rbt6_L-6_H-768_A-12/vocab.txt --model_type bert --gpu 2 --batch_size 256 --freeze &&
#Epoch 10/10000
#188/188 - 50s - loss: 1.4343 - accuracy: 0.5263
#dev acc=53.41%
#test acc=53.85%

python -u run_fine_tuning_bert4keras_models.py --config_path /data0/nfs_data/zhaoxi9/pretrained_language_model/nezha/NEZHA-Large-WWM/bert_config.json --checkpoint_path /data0/nfs_data/zhaoxi9/pretrained_language_model/nezha/NEZHA-Large-WWM/model.ckpt-346400 --vocab_path /data0/nfs_data/zhaoxi9/pretrained_language_model/nezha/NEZHA-Large-WWM/vocab.txt --model_type nezha --gpu 2 --batch_size 256 --freeze &&
#Epoch 27/10000
#188/188 - 310s - loss: 1.3686 - accuracy: 0.5409
#dev acc=56.09%
#test acc=55.59%
 python -u run_fine_tuning_bert4keras_models.py --config_path /data0/nfs_data/zhaoxi9/pretrained_language_model/nezha/NEZHA-Large/bert_config.json --checkpoint_path /data0/nfs_data/zhaoxi9/pretrained_language_model/nezha/NEZHA-Large/model.ckpt-325810 --vocab_path /data0/nfs_data/zhaoxi9/pretrained_language_model/nezha/NEZHA-Large/vocab.txt --model_type nezha --gpu 2 --batch_size 256 --freeze &&
#Epoch 22/10000
#188/188 - 311s - loss: 1.3472 - accuracy: 0.5498
#dev acc=56.35%
#test acc=55.86%

python -u run_fine_tuning_bert4keras_models.py --config_path /data0/nfs_data/zhaoxi9/pretrained_language_model/nezha/NEZHA-Base-WWM/bert_config.json --checkpoint_path /data0/nfs_data/zhaoxi9/pretrained_language_model/nezha/NEZHA-Base-WWM/model.ckpt-691689 --vocab_path /data0/nfs_data/zhaoxi9/pretrained_language_model/nezha/NEZHA-Base-WWM/vocab.txt --model_type nezha --gpu 2 --batch_size 256 --freeze &&
#Epoch 16/10000
#188/188 - 105s - loss: 1.3953 - accuracy: 0.5395
#dev acc=56.07%
#test acc=56.16%
python -u run_fine_tuning_bert4keras_models.py --config_path /data0/nfs_data/zhaoxi9/pretrained_language_model/nezha/NEZHA-Base/bert_config.json --checkpoint_path /data0/nfs_data/zhaoxi9/pretrained_language_model/nezha/NEZHA-Base/model.ckpt-900000 --vocab_path /data0/nfs_data/zhaoxi9/pretrained_language_model/nezha/NEZHA-Base/vocab.txt --model_type nezha --gpu 2 --batch_size 256 --freeze
#Epoch 13/10000
#188/188 - 105s - loss: 1.4215 - accuracy: 0.5317
#dev acc=55.38%
#test acc=55.14%