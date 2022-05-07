#!/bin/bash
python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/junnyu_roformer_chinese_sim_char_ft_small --gpu 3 --freeze --batch_size 2048 &&
#epoch: 36/10000, 15s, train loss=1.8841, train acc=40.06%, dev loss=1.6555, dev acc=47.98%
#saving, test loss=1.6514, test acc=48.87%
python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/luhua_chinese_pretrain_mrc_macbert_large --gpu 3 --freeze --batch_size 2048 &&
#epoch: 34/10000, 225s, train loss=1.5641, train acc=49.58%, dev loss=1.3997, dev acc=53.45%
#saving, test loss=1.4047, test acc=53.52%
python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/luhua_chinese_pretrain_mrc_roberta_wwm_ext_large --gpu 3 --freeze --batch_size 2048 &&
#epoch: 23/10000, 225s, train loss=1.7745, train acc=43.61%, dev loss=1.5530, dev acc=50.97%
#saving, test loss=1.5695, test acc=49.56%
python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/peterchou_nezha-chinese-base --gpu 3 --freeze --batch_size 2048 &&
#epoch: 4/10000, 90s, train loss=2.5374, train acc=14.57%, dev loss=2.5134, dev acc=16.19%
#saving, test loss=2.5189, test acc=15.36%
python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/uer_chinese_roberta_L-4_H-512 --gpu 3 --freeze --batch_size 2048 &&
#epoch: 41/10000, 17s, train loss=1.5554, train acc=49.04%, dev loss=1.4479, dev acc=51.67%
#saving, test loss=1.4535, test acc=51.91%
python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/uer_roberta-base-finetuned-jd-full-chinese --gpu 3 --freeze --batch_size 2048
#epoch: 70/10000, 70s, train loss=1.7754, train acc=43.68%, dev loss=1.6196, dev acc=48.05%
#saving, test loss=1.6288, test acc=47.85%