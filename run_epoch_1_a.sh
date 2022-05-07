#!/bin/bash
python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/bert-base-chinese --gpu 0 --freeze --batch_size 2048 &&
#epoch: 22/10000, 70s, train loss=1.3635, train acc=54.04%, dev loss=1.2922, dev acc=54.61%
#saving, test loss=1.3079, test acc=54.54%
python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/ckiplab_albert-tiny-chinese --gpu 0 --freeze --batch_size 2048 &&
#epoch: 51/10000, 7s, train loss=2.0111, train acc=34.80%, dev loss=1.8673, dev acc=40.74%
#saving, test loss=1.8926, test acc=39.48%
python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/ckiplab_bert-base-chinese --gpu 0 --freeze --batch_size 2048 &&
#epoch: 20/10000, 70s, train loss=1.4237, train acc=52.62%, dev loss=1.3267, dev acc=54.07%
#saving, test loss=1.3419, test acc=53.96%
python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/clue_albert_chinese_small --gpu 0 --freeze --batch_size 2048 &&
#epoch: 36/10000, 12s, train loss=1.6859, train acc=45.63%, dev loss=1.5814, dev acc=47.38%
#saving, test loss=1.5988, test acc=48.07%
python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/clue_albert_chinese_tiny --gpu 0 --freeze --batch_size 2048 &&
#epoch: 44/10000, 7s, train loss=1.7558, train acc=43.33%, dev loss=1.6301, dev acc=46.63%
#saving, test loss=1.6531, test acc=47.08%
python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/clue_roberta_chinese_3L312_clue_tiny --gpu 0 --freeze --batch_size 2048 &&
#epoch: 50/10000, 6s, train loss=2.5092, train acc=16.22%, dev loss=2.4557, dev acc=21.35%
#saving, test loss=2.4565, test acc=21.19%
python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/clue_roberta_chinese_3L768_clue_tiny --gpu 0 --freeze --batch_size 2048 &&
#epoch: 40/10000, 18s, train loss=2.3911, train acc=22.12%, dev loss=2.2810, dev acc=28.86%
#saving, test loss=2.2757, test acc=29.94%
python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/clue_roberta_chinese_base --gpu 0 --freeze --batch_size 2048 &&
#epoch: 35/10000, 70s, train loss=2.4209, train acc=20.73%, dev loss=2.3142, dev acc=26.05%
#saving, test loss=2.3189, test acc=25.51%
python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/clue_roberta_chinese_clue_large --gpu 0 --freeze --batch_size 2048 &&
#epoch: 23/10000, 230s, train loss=2.4739, train acc=18.32%, dev loss=2.3467, dev acc=25.34%
#saving, test loss=2.3533, test acc=24.97%
python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/clue_roberta_chinese_clue_tiny --gpu 0 --freeze --batch_size 2048 &&
#epoch: 41/10000, 12s, train loss=2.4909, train acc=17.41%, dev loss=2.4362, dev acc=22.88%
#saving, test loss=2.4381, test acc=22.21%
python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/clue_roberta_chinese_large --gpu 0 --freeze --batch_size 2048 &&
#epoch: 32/10000, 225s, train loss=2.4734, train acc=18.23%, dev loss=2.3567, dev acc=24.63%
#saving, test loss=2.3619, test acc=24.19%
python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/clue_roberta_chinese_pair_large --gpu 0 --freeze --batch_size 2048 &&
#epoch: 21/10000, 229s, train loss=2.4674, train acc=18.56%, dev loss=2.3474, dev acc=25.26%
#saving, test loss=2.3533, test acc=25.03%
python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/clue_roberta_chinese_pair_tiny --gpu 0 --freeze --batch_size 2048 &&
#epoch: 34/10000, 7s, train loss=2.4925, train acc=17.33%, dev loss=2.4465, dev acc=22.60%
#saving, test loss=2.4484, test acc=22.06%
python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/clue_xlnet_chinese_large --gpu 0 --freeze --batch_size 2048
#epoch: 19/10000, 241s, train loss=1.8296, train acc=40.79%, dev loss=1.6497, dev acc=45.99%
#saving, test loss=1.6573, test acc=46.34%

# clue_albert_chinese_tiny, 16M，很强，看下载量就看得出来，其他下载量都不行
# clue_albert_chinese_small，19M,也很强，但是模型比tiny大
# clue_roberta_chinese_base，下载量也不小，但是效果不行
