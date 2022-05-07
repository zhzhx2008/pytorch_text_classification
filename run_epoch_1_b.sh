#!/bin/bash
python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/hfl_chinese-bert-wwm --gpu 1 --freeze --batch_size 2048 &&
#epoch: 12/10000, 70s, train loss=1.3914, train acc=53.65%, dev loss=1.3103, dev acc=54.50%
#saving, test loss=1.3272, test acc=54.87%
python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/hfl_chinese-bert-wwm-ext --gpu 1 --freeze --batch_size 2048 &&
#epoch: 23/10000, 71s, train loss=1.3433, train acc=54.78%, dev loss=1.2709, dev acc=55.53%
#saving, test loss=1.2857, test acc=55.58%
python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/hfl_chinese-electra-180g-base-discriminator --gpu 1 --freeze --batch_size 2048 &&
#epoch: 53/10000, 71s, train loss=2.1869, train acc=28.95%, dev loss=2.0858, dev acc=33.86%
#saving, test loss=2.0827, test acc=33.71%
python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/hfl_chinese-electra-180g-large-discriminator --gpu 1 --freeze --batch_size 2048 &&
#epoch: 37/10000, 294s, train loss=2.1949, train acc=28.96%, dev loss=2.0709, dev acc=35.10%
#saving, test loss=2.0716, test acc=34.78%
python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/hfl_chinese-electra-180g-small-discriminator --gpu 1 --freeze --batch_size 2048 &&
#epoch: 76/10000, 21s, train loss=2.3397, train acc=23.47%, dev loss=2.2167, dev acc=29.59%
#saving, test loss=2.2252, test acc=28.53%
python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/hfl_chinese-electra-180g-small-ex-discriminator --gpu 1 --freeze --batch_size 2048 &&
#epoch: 55/10000, 30s, train loss=2.3457, train acc=23.29%, dev loss=2.2638, dev acc=27.70%
#saving, test loss=2.2590, test acc=27.20%
python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/hfl_chinese-electra-base-discriminator --gpu 1 --freeze --batch_size 2048 &&
#epoch: 38/10000, 71s, train loss=2.2722, train acc=26.06%, dev loss=2.2172, dev acc=27.87%
#saving, test loss=2.2227, test acc=27.51%
python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/hfl_chinese-electra-base-generator --gpu 1 --freeze --batch_size 2048 &&
#epoch: 89/10000, 12s, train loss=2.0271, train acc=35.29%, dev loss=1.8205, dev acc=46.48%
#saving, test loss=1.8228, test acc=46.80%
python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/hfl_chinese-electra-large-discriminator --gpu 1 --freeze --batch_size 2048 &&
#epoch: 41/10000, 226s, train loss=2.4223, train acc=20.16%, dev loss=2.3690, dev acc=23.95%
#saving, test loss=2.3742, test acc=22.59%
python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/hfl_chinese-electra-large-generator --gpu 1 --freeze --batch_size 2048 &&
#epoch: 37/10000, 30s, train loss=1.8497, train acc=40.93%, dev loss=1.6242, dev acc=49.12%
#saving, test loss=1.6276, test acc=49.80%
python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/hfl_chinese-electra-small-ex-discriminator --gpu 1 --freeze --batch_size 2048 &&
#epoch: 67/10000, 29s, train loss=2.3450, train acc=23.35%, dev loss=2.2592, dev acc=26.59%
#saving, test loss=2.2563, test acc=26.04%
python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/hfl_chinese-electra-small-ex-generator --gpu 1 --freeze --batch_size 2048
#epoch: 78/10000, 9s, train loss=2.1604, train acc=29.76%, dev loss=1.9463, dev acc=40.05%
#saving, test loss=1.9498, test acc=40.01%

# 还是bert最强