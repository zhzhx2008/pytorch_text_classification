#!/bin/bash
python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/hfl_chinese-legal-electra-base-generator --gpu 2 --freeze --batch_size 2048 &&
#epoch: 83/10000, 12s, train loss=2.0694, train acc=33.94%, dev loss=1.8819, dev acc=43.74%
#saving, test loss=1.8885, test acc=43.74%
python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/hfl_chinese-legal-electra-large-generator --gpu 2 --freeze --batch_size 2048 &&
#epoch: 47/10000, 30s, train loss=1.9414, train acc=37.48%, dev loss=1.7075, dev acc=47.62%
#saving, test loss=1.7152, test acc=47.39%
python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/hfl_chinese-legal-electra-small-discriminator --gpu 2 --freeze --batch_size 2048 &&
#epoch: 44/10000, 15s, train loss=2.3386, train acc=23.56%, dev loss=2.2810, dev acc=25.82%
#saving, test loss=2.2834, test acc=25.99%
python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/hfl_chinese-legal-electra-small-generator --gpu 2 --freeze --batch_size 2048 &&
#epoch: 57/10000, 5s, train loss=2.4396, train acc=19.12%, dev loss=2.3595, dev acc=24.66%
#saving, test loss=2.3566, test acc=24.25%
python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/hfl_chinese-macbert-base --gpu 2 --freeze --batch_size 2048 &&
#epoch: 20/10000, 71s, train loss=1.3524, train acc=54.34%, dev loss=1.2708, dev acc=55.73%
#saving, test loss=1.2756, test acc=55.85%
python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/hfl_chinese-macbert-large --gpu 2 --freeze --batch_size 2048 &&
#epoch: 32/10000, 223s, train loss=1.3273, train acc=55.07%, dev loss=1.2772, dev acc=56.33%
#saving, test loss=1.2749, test acc=55.77%
python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/hfl_chinese-pert-base --gpu 2 --freeze --batch_size 2048 &&
#epoch: 105/10000, 71s, train loss=1.5039, train acc=51.71%, dev loss=1.4225, dev acc=53.64%
#saving, test loss=1.4259, test acc=53.83%
python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/hfl_chinese-pert-large --gpu 2 --freeze --batch_size 2048 &&
#epoch: 140/10000, 229s, train loss=2.0741, train acc=33.47%, dev loss=1.9797, dev acc=38.59%
#saving, test loss=1.9715, test acc=38.83%
python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/hfl_chinese-roberta-wwm-ext --gpu 2 --freeze --batch_size 2048 &&
#epoch: 15/10000, 71s, train loss=1.3226, train acc=55.16%, dev loss=1.2494, dev acc=56.52%
#saving, test loss=1.2625, test acc=55.95%
python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/hfl_chinese-roberta-wwm-ext-large --gpu 2 --freeze --batch_size 2048 &&
#epoch: 19/10000, 226s, train loss=1.2979, train acc=56.04%, dev loss=1.2474, dev acc=56.82%
#saving, test loss=1.2513, test acc=56.21%
python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/hfl_chinese-xlnet-base --gpu 2 --freeze --batch_size 2048 &&
#epoch: 12/10000, 82s, train loss=1.8170, train acc=42.26%, dev loss=1.6623, dev acc=47.98%
#saving, test loss=1.6615, test acc=48.02%
python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/hfl_chinese-xlnet-mid --gpu 2 --freeze --batch_size 2048 &&
#epoch: 10/10000, 159s, train loss=2.0526, train acc=36.09%, dev loss=1.8179, dev acc=44.43%
#saving, test loss=1.8269, test acc=43.17%
python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/hfl_rbt3 --gpu 2 --freeze --batch_size 2048 &&
#epoch: 11/10000, 18s, train loss=1.5268, train acc=50.32%, dev loss=1.4485, dev acc=51.87%
#saving, test loss=1.4612, test acc=52.01%
python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/hfl_rbt4 --gpu 2 --freeze --batch_size 2048 &&
#epoch: 13/10000, 24s, train loss=1.5020, train acc=51.20%, dev loss=1.4202, dev acc=53.19%
#saving, test loss=1.4253, test acc=52.98%
python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/hfl_rbt6 --gpu 2 --freeze --batch_size 2048 &&
#epoch: 20/10000, 36s, train loss=1.4581, train acc=52.00%, dev loss=1.3720, dev acc=53.17%
#saving, test loss=1.3750, test acc=53.81%
python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/hfl_rbtl3 --gpu 2 --freeze --batch_size 2048
#epoch: 20/10000, 28s, train loss=1.4464, train acc=51.89%, dev loss=1.3830, dev acc=53.15%
#saving, test loss=1.4001, test acc=53.39%




nohup python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/hfl_chinese-roberta-wwm-ext-large --gpu 2 --batch_size 128 > nohup_hfl_chinese-roberta-wwm-ext-large.out 2>&1 &
