#!/bin/bash
python -u run_fine_tuning_bert4keras_models.py --config_path /data0/nfs_data/zhaoxi9/pretrained_language_model/Chinese-BERT-wwm/chinese_L-12_H-768_A-12/bert_config.json --checkpoint_path /data0/nfs_data/zhaoxi9/pretrained_language_model/Chinese-BERT-wwm/chinese_L-12_H-768_A-12/bert_model.ckpt --vocab_path /data0/nfs_data/zhaoxi9/pretrained_language_model/Chinese-BERT-wwm/chinese_L-12_H-768_A-12/vocab.txt --model_type bert --gpu 0 --batch_size 64 &&
#Epoch 1/10000
#751/751 - 334s - loss: 2.6585 - accuracy: 0.0943
#dev acc=11.15%
#test acc=10.89%
python -u run_fine_tuning_bert4keras_models.py --config_path /data0/nfs_data/zhaoxi9/pretrained_language_model/Chinese-BERT-wwm/chinese_bert_wwm_L-12_H-768_A-12/publish/bert_config.json --checkpoint_path /data0/nfs_data/zhaoxi9/pretrained_language_model/Chinese-BERT-wwm/chinese_bert_wwm_L-12_H-768_A-12/publish/bert_model.ckpt --vocab_path /data0/nfs_data/zhaoxi9/pretrained_language_model/Chinese-BERT-wwm/chinese_bert_wwm_L-12_H-768_A-12/publish/vocab.txt --model_type bert --gpu 0 --batch_size 64 &&
#Epoch 2/10000
#751/751 - 305s - loss: 2.6131 - accuracy: 0.1044
#dev acc=11.15%
#test acc=10.89%
python -u run_fine_tuning_bert4keras_models.py --config_path /data0/nfs_data/zhaoxi9/pretrained_language_model/Chinese-BERT-wwm/chinese_rbt4_L-4_H-768_A-12/bert_config_rbt4.json --checkpoint_path /data0/nfs_data/zhaoxi9/pretrained_language_model/Chinese-BERT-wwm/chinese_rbt4_L-4_H-768_A-12/bert_model.ckpt --vocab_path /data0/nfs_data/zhaoxi9/pretrained_language_model/Chinese-BERT-wwm/chinese_rbt4_L-4_H-768_A-12/vocab.txt --model_type bert --gpu 0 --batch_size 64 &&
#Epoch 1/10000
#751/751 - 124s - loss: 2.2816 - accuracy: 0.2569
#dev acc=28.75%
#test acc=27.68%
python -u run_fine_tuning_bert4keras_models.py --config_path /data0/nfs_data/zhaoxi9/pretrained_language_model/Chinese-BERT-wwm/chinese_rbt6_L-6_H-768_A-12/bert_config_rbt6.json --checkpoint_path /data0/nfs_data/zhaoxi9/pretrained_language_model/Chinese-BERT-wwm/chinese_rbt6_L-6_H-768_A-12/bert_model.ckpt --vocab_path /data0/nfs_data/zhaoxi9/pretrained_language_model/Chinese-BERT-wwm/chinese_rbt6_L-6_H-768_A-12/vocab.txt --model_type bert --gpu 0 --batch_size 64 &&
#Epoch 1/10000
#751/751 - 180s - loss: 2.6204 - accuracy: 0.1029
#dev acc=11.15%
#test acc=10.89%

python -u run_fine_tuning_bert4keras_models.py --config_path /data0/nfs_data/zhaoxi9/pretrained_language_model/nezha/NEZHA-Large-WWM/bert_config.json --checkpoint_path /data0/nfs_data/zhaoxi9/pretrained_language_model/nezha/NEZHA-Large-WWM/model.ckpt-346400 --vocab_path /data0/nfs_data/zhaoxi9/pretrained_language_model/nezha/NEZHA-Large-WWM/vocab.txt --model_type nezha --gpu 0 --batch_size 64 &&
#Epoch 1/10000
#751/751 - 992s - loss: 2.6171 - accuracy: 0.1029
#dev acc=11.15%
#test acc=10.89%
python -u run_fine_tuning_bert4keras_models.py --config_path /data0/nfs_data/zhaoxi9/pretrained_language_model/nezha/NEZHA-Large/bert_config.json --checkpoint_path /data0/nfs_data/zhaoxi9/pretrained_language_model/nezha/NEZHA-Large/model.ckpt-325810 --vocab_path /data0/nfs_data/zhaoxi9/pretrained_language_model/nezha/NEZHA-Large/vocab.txt --model_type nezha --gpu 0 --batch_size 64 &&
#Epoch 1/10000
#751/751 - 995s - loss: 2.6187 - accuracy: 0.1027
#dev acc=11.15%
#test acc=10.89%
python -u run_fine_tuning_bert4keras_models.py --config_path /data0/nfs_data/zhaoxi9/pretrained_language_model/nezha/NEZHA-Base-WWM/bert_config.json --checkpoint_path /data0/nfs_data/zhaoxi9/pretrained_language_model/nezha/NEZHA-Base-WWM/model.ckpt-691689 --vocab_path /data0/nfs_data/zhaoxi9/pretrained_language_model/nezha/NEZHA-Base-WWM/vocab.txt --model_type nezha --gpu 0 --batch_size 64 &&
#Epoch 1/10000
#751/751 - 343s - loss: 2.6149 - accuracy: 0.1039
#dev acc=11.15%
#test acc=10.89%
python -u run_fine_tuning_bert4keras_models.py --config_path /data0/nfs_data/zhaoxi9/pretrained_language_model/nezha/NEZHA-Base/bert_config.json --checkpoint_path /data0/nfs_data/zhaoxi9/pretrained_language_model/nezha/NEZHA-Base/model.ckpt-900000 --vocab_path /data0/nfs_data/zhaoxi9/pretrained_language_model/nezha/NEZHA-Base/vocab.txt --model_type nezha --gpu 0 --batch_size 64
#Epoch 1/10000
#751/751 - 343s - loss: 2.6190 - accuracy: 0.1002
#dev acc=11.15%
#test acc=10.89%