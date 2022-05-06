#!/bin/bash
python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/hfl_chinese-legal-electra-base-generator --gpu 2 --freeze --batch_size 2048 &&
python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/hfl_chinese-legal-electra-large-generator --gpu 2 --freeze --batch_size 2048 &&
python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/hfl_chinese-legal-electra-small-discriminator --gpu 2 --freeze --batch_size 2048 &&
python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/hfl_chinese-legal-electra-small-generator --gpu 2 --freeze --batch_size 2048 &&
python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/hfl_chinese-macbert-base --gpu 2 --freeze --batch_size 2048 &&
python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/hfl_chinese-macbert-large --gpu 2 --freeze --batch_size 2048 &&
python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/hfl_chinese-pert-base --gpu 2 --freeze --batch_size 2048 &&
python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/hfl_chinese-pert-large --gpu 2 --freeze --batch_size 2048 &&
python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/hfl_chinese-roberta-wwm-ext --gpu 2 --freeze --batch_size 2048 &&
python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/hfl_chinese-roberta-wwm-ext-large --gpu 3 --freeze --batch_size 2048 &&
python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/hfl_chinese-xlnet-base --gpu 2 --freeze --batch_size 2048 &&
python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/hfl_chinese-xlnet-mid --gpu 2 --freeze --batch_size 2048 &&
python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/hfl_rbt3 --gpu 2 --freeze --batch_size 2048 &&
python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/hfl_rbt4 --gpu 2 --freeze --batch_size 2048 &&
python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/hfl_rbt6 --gpu 2 --freeze --batch_size 2048 &&
python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/hfl_rbtl3 --gpu 2 --freeze --batch_size 2048