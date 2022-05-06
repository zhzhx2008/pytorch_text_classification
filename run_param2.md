FastText dropout 0.2
python -u run.py --ngrams_char 1 --min_freq_char 1 --batch_size 32 --learning_rate 2e-4 --gpu 1 --model_name fasttext	dev acc=51.67%	test acc=50.65%
python -u run.py --ngrams_char 1 --min_freq_char 2 --batch_size 32 --learning_rate 2e-4 --gpu 1 --model_name fasttext	dev acc=51.71%	test acc=50.55%
python -u run.py --ngrams_char 1 --min_freq_char 3 --batch_size 32 --learning_rate 2e-4 --gpu 1 --model_name fasttext	dev acc=51.37%	test acc=50.59%
python -u run.py --ngrams_char 1 --min_freq_char 4 --batch_size 32 --learning_rate 2e-4 --gpu 1 --model_name fasttext	dev acc=51.65%	test acc=50.72%
python -u run.py --ngrams_char 1 --min_freq_char 5 --batch_size 32 --learning_rate 2e-4 --gpu 1 --model_name fasttext	dev acc=51.59%	test acc=50.66%
python -u run.py --ngrams_char 1 --min_freq_char 6 --batch_size 32 --learning_rate 2e-4 --gpu 1 --model_name fasttext	dev acc=51.84%	test acc=50.72%
python -u run.py --ngrams_char 1 --min_freq_char 7 --batch_size 32 --learning_rate 2e-4 --gpu 1 --model_name fasttext	dev acc=51.54%	test acc=50.59%
python -u run.py --ngrams_char 1 --min_freq_char 8 --batch_size 32 --learning_rate 2e-4 --gpu 1 --model_name fasttext	dev acc=51.44%	test acc=50.70%
python -u run.py --ngrams_char 1 --min_freq_char 9 --batch_size 32 --learning_rate 2e-4 --gpu 1 --model_name fasttext	dev acc=51.31%	test acc=50.38%
python -u run.py --ngrams_char 2 --min_freq_char 1 --batch_size 32 --learning_rate 2e-4 --gpu 1 --model_name fasttext	dev acc=46.95%	test acc=47.70%
python -u run.py --ngrams_char 2 --min_freq_char 2 --batch_size 32 --learning_rate 2e-4 --gpu 1 --model_name fasttext	dev acc=48.73%	test acc=49.22%
python -u run.py --ngrams_char 2 --min_freq_char 3 --batch_size 32 --learning_rate 2e-4 --gpu 1 --model_name fasttext	dev acc=48.78%	test acc=49.71%
python -u run.py --ngrams_char 2 --min_freq_char 4 --batch_size 32 --learning_rate 2e-4 --gpu 1 --model_name fasttext	dev acc=48.67%	test acc=50.28%
python -u run.py --ngrams_char 2 --min_freq_char 5 --batch_size 32 --learning_rate 2e-4 --gpu 1 --model_name fasttext	dev acc=49.04%	test acc=50.35%
python -u run.py --ngrams_char 2 --min_freq_char 6 --batch_size 32 --learning_rate 2e-4 --gpu 1 --model_name fasttext	dev acc=48.93%	test acc=50.52%
python -u run.py --ngrams_char 2 --min_freq_char 7 --batch_size 32 --learning_rate 2e-4 --gpu 1 --model_name fasttext   dev acc=49.38%  test acc=50.53%
python -u run.py --ngrams_char 2 --min_freq_char 8 --batch_size 32 --learning_rate 2e-4 --gpu 1 --model_name fasttext   dev acc=49.23%  test acc=50.35%
python -u run.py --ngrams_char 2 --min_freq_char 9 --batch_size 32 --learning_rate 2e-4 --gpu 1 --model_name fasttext   dev acc=49.79%  test acc=49.98%
python -u run.py --ngrams_char 2 --min_freq_char 10 --batch_size 32 --learning_rate 2e-4 --gpu 1 --model_name fasttext   dev acc=49.59% test acc=50.45%
python -u run.py --ngrams_char 2 --min_freq_char 11 --batch_size 32 --learning_rate 2e-4 --gpu 1 --model_name fasttext   dev acc=49.36% test acc=49.96%
python -u run.py --ngrams_char 2 --min_freq_char 12 --batch_size 32 --learning_rate 2e-4 --gpu 1 --model_name fasttext   dev acc=49.34% test acc=49.78%

python -u run.py --ngrams_word 1 --min_freq_word 1 --batch_size 32 --learning_rate 2e-4 --gpu 1 --model_name fasttext	dev acc=48.50%	test acc=49.12%
python -u run.py --ngrams_word 1 --min_freq_word 2 --batch_size 32 --learning_rate 2e-4 --gpu 1 --model_name fasttext	dev acc=50.37%	test acc=50.75%
python -u run.py --ngrams_word 1 --min_freq_word 3 --batch_size 32 --learning_rate 2e-4 --gpu 1 --model_name fasttext	dev acc=50.58%	test acc=50.39%
python -u run.py --ngrams_word 1 --min_freq_word 4 --batch_size 32 --learning_rate 2e-4 --gpu 1 --model_name fasttext	dev acc=50.97%	test acc=50.79%
python -u run.py --ngrams_word 1 --min_freq_word 5 --batch_size 32 --learning_rate 2e-4 --gpu 1 --model_name fasttext	dev acc=50.81%	test acc=50.79%
python -u run.py --ngrams_word 1 --min_freq_word 6 --batch_size 32 --learning_rate 2e-4 --gpu 1 --model_name fasttext	dev acc=50.67%	 test acc=50.75%
python -u run.py --ngrams_word 2 --min_freq_word 1 --batch_size 32 --learning_rate 2e-4 --gpu 1 --model_name fasttext	dev acc=26.80%	test acc=26.76%
python -u run.py --ngrams_word 2 --min_freq_word 2 --batch_size 32 --learning_rate 2e-4 --gpu 1 --model_name fasttext	dev acc=38.12%	test acc=36.78%
python -u run.py --ngrams_word 2 --min_freq_word 3 --batch_size 32 --learning_rate 2e-4 --gpu 1 --model_name fasttext	dev acc=36.38%	test acc=35.74%
python -u run.py --ngrams_word 2 --min_freq_word 4 --batch_size 32 --learning_rate 2e-4 --gpu 1 --model_name fasttext	dev acc=35.25%	test acc=34.46%
python -u run.py --ngrams_word 2 --min_freq_word 5 --batch_size 32 --learning_rate 2e-4 --gpu 1 --model_name fasttext	dev acc=33.88%	test acc=33.65%
python -u run.py --ngrams_word 2 --min_freq_word 6 --batch_size 32 --learning_rate 2e-4 --gpu 1 --model_name fasttext	dev acc=32.12%	test acc=32.34%

python -u run.py --ngrams_char 1 --min_freq_char 6 --ngrams_word 1 --min_freq_word 4 --batch_size 32 --learning_rate 2e-4 --gpu 1 --model_name fasttext	dev acc=52.79%	test acc=52.06%
python -u run.py --ngrams_char 1 2 --min_freq_char 6 3 --batch_size 32 --learning_rate 2e-4 --gpu 1 --model_name fasttext	dev acc=52.72%	 test acc=51.98%
python -u run.py --ngrams_char 1 2 --min_freq_char 6 9 --batch_size 32 --learning_rate 2e-4 --gpu 1 --model_name fasttext	dev acc=52.38%	 test acc=52.43%
python -u run.py --ngrams_word 1 2 --min_freq_word 4 2 --batch_size 32 --learning_rate 2e-4 --gpu 1 --model_name fasttext	dev acc=49.21%	test acc=49.00%
python -u run.py --ngrams_char 1 2 --min_freq_char 6 3 --ngrams_word 1 2 --min_freq_word 4 2 --batch_size 32 --learning_rate 2e-4 --gpu 1 --model_name fasttext	dev acc=52.74%	test acc=52.17%
python -u run.py --ngrams_char 1 2 --min_freq_char 6 9 --ngrams_word 1 2 --min_freq_word 4 2 --batch_size 32 --learning_rate 2e-4 --gpu 1 --model_name fasttext	dev acc=52.64%	test acc=52.68%



FastText dropout 0.5
python -u run.py --ngrams_char 1 --min_freq_char 1 --batch_size 32 --learning_rate 2e-4 --gpu 1 --model_name fasttext	dev acc=51.91%	test acc=51.21%
python -u run.py --ngrams_char 1 --min_freq_char 2 --batch_size 32 --learning_rate 2e-4 --gpu 1 --model_name fasttext	dev acc=51.80%	test acc=51.13%
python -u run.py --ngrams_char 1 --min_freq_char 6 --batch_size 32 --learning_rate 2e-4 --gpu 1 --model_name fasttext	dev acc=51.78%	test acc=50.86%
python -u run.py --ngrams_char 1 --min_freq_char 6 --ngrams_word 1 --min_freq_word 4 --batch_size 32 --learning_rate 2e-4 --gpu 1 --model_name fasttext 	dev acc=53.30%	test acc=52.53%
python -u run.py --ngrams_char 1 2 --min_freq_char 6 3 --ngrams_word 1 2 --min_freq_word 4 2 --batch_size 32 --learning_rate 2e-4 --gpu 1 --model_name fasttext	dev acc=53.54%	test acc=53.00%
python -u run.py --ngrams_char 1 2 --min_freq_char 6 9 --ngrams_word 1 2 --min_freq_word 4 2 --batch_size 32 --learning_rate 2e-4 --gpu 1 --model_name fasttext	dev acc=53.15%	test acc=52.35%



FastText dropout 0.7
python -u run.py --ngrams_char 1 --min_freq_char 6 --ngrams_word 1 --min_freq_word 4 --batch_size 32 --learning_rate 2e-4 --gpu 1 --model_name fasttext 	dev acc=53.92%	test acc=52.78%
python -u run.py --ngrams_char 1 2 --min_freq_char 6 3 --ngrams_word 1 2 --min_freq_word 4 2 --batch_size 32 --learning_rate 2e-4 --gpu 1 --model_name fasttext --dropout 0.7   dev acc=53.39%	test acc=52.35%


FastText embedding
python -u run.py --ngrams_char 1 --min_freq_char 6 --batch_size 32 --learning_rate 2e-4 --dropout 0.2 --gpu 1 --model_name fasttext --embedding_file /data0/nfs_data/zhaoxi9/pretrained_language_model/Chinese-Word-Vectors/merge_sgns_bigram_char300.txt   dev acc=51.61%	test acc=51.09%
python -u run.py --ngrams_word 1 --min_freq_word 4 --batch_size 32 --learning_rate 2e-4 --dropout 0.2 --gpu 1 --model_name fasttext --embedding_file /data0/nfs_data/zhaoxi9/pretrained_language_model/Chinese-Word-Vectors/merge_sgns_bigram_char300.txt   dev acc=53.92%	test acc=54.46%
python -u run.py --ngrams_word 1 --min_freq_word 4 --batch_size 32 --learning_rate 2e-4 --dropout 0.2 --gpu 1 --model_name fasttext --embedding_file /data0/nfs_data/zhaoxi9/pretrained_language_model/Chinese-Word-Vectors/merge_sgns_bigram_char300.txt --padding_idx   dev acc=54.39%	test acc=55.47%
python -u run.py --ngrams_word 1 --min_freq_word 4 --batch_size 32 --learning_rate 2e-4 --dropout 0.5 --gpu 1 --model_name fasttext --embedding_file /data0/nfs_data/zhaoxi9/pretrained_language_model/Chinese-Word-Vectors/merge_sgns_bigram_char300.txt --padding_idx   dev acc=54.52%	test acc=55.33%
python -u run.py --ngrams_word 1 --min_freq_word 4 --batch_size 32 --learning_rate 2e-4 --dropout 0.7 --gpu 1 --model_name fasttext --embedding_file /data0/nfs_data/zhaoxi9/pretrained_language_model/Chinese-Word-Vectors/merge_sgns_bigram_char300.txt --padding_idx   dev acc=54.39%	test acc=55.13%

python -u run.py --ngrams_word 1 --min_freq_word 4 --batch_size 32 --learning_rate 2e-4 --dropout 0.5 --gpu 0 --model_name fasttext --embedding_file /data0/nfs_data/zhaoxi9/pretrained_language_model/Chinese-Word-Vectors/sgns.merge.bigram --padding_idx   dev acc=54.22%	test acc=55.03%
python -u run.py --ngrams_word 1 --min_freq_word 4 --batch_size 32 --learning_rate 2e-4 --dropout 0.5 --gpu 0 --model_name fasttext --embedding_file /data0/nfs_data/zhaoxi9/pretrained_language_model/Chinese-Word-Vectors/sgns.merge.char --padding_idx   dev acc=54.12%	test acc=54.71%
python -u run.py --ngrams_word 1 --min_freq_word 4 --batch_size 32 --learning_rate 2e-4 --dropout 0.5 --gpu 0 --model_name fasttext --embedding_file /data0/nfs_data/zhaoxi9/pretrained_language_model/Chinese-Word-Vectors/sgns.merge.word --padding_idx   dev acc=54.09%	test acc=54.74%
python -u run.py --ngrams_word 1 --min_freq_word 4 --batch_size 32 --learning_rate 2e-4 --dropout 0.5 --gpu 1 --model_name fasttext --embedding_file /data0/nfs_data/zhaoxi9/pretrained_language_model/Chinese-Word-Vectors/sgns.sogou.bigram --padding_idx   dev acc=53.82%	test acc=54.52%
python -u run.py --ngrams_word 1 --min_freq_word 4 --batch_size 32 --learning_rate 2e-4 --dropout 0.5 --gpu 1 --model_name fasttext --embedding_file /data0/nfs_data/zhaoxi9/pretrained_language_model/Chinese-Word-Vectors/sgns.sogou.char --padding_idx   dev acc=53.65%	test acc=54.75%
python -u run.py --ngrams_word 1 --min_freq_word 4 --batch_size 32 --learning_rate 2e-4 --dropout 0.5 --gpu 1 --model_name fasttext --embedding_file /data0/nfs_data/zhaoxi9/pretrained_language_model/Chinese-Word-Vectors/sgns.sogou.word --padding_idx   dev acc=53.80%	test acc=54.71%
python -u run.py --ngrams_word 1 --min_freq_word 4 --batch_size 32 --learning_rate 2e-4 --dropout 0.5 --gpu 1 --model_name fasttext --embedding_file /data0/nfs_data/zhaoxi9/pretrained_language_model/Tencent_AILab_ChineseEmbedding.txt --padding_idx   dev acc=54.14%	test acc=54.52%
python -u run.py --ngrams_word 1 --min_freq_word 4 --batch_size 32 --learning_rate 2e-4 --dropout 0.5 --gpu 0 --model_name fasttext --embedding_file /data0/nfs_data/zhaoxi9/pretrained_language_model/baike_26g_news_13g_novel_229g.bin --padding_idx   dev acc=54.22%	test acc=53.69%

/data0/nfs_data/zhaoxi9/pretrained_language_model/Chinese-Word-Vectors/merge_sgns_bigram_char300.txt
/data0/nfs_data/zhaoxi9/pretrained_language_model/Chinese-Word-Vectors/sgns.merge.bigram
/data0/nfs_data/zhaoxi9/pretrained_language_model/Chinese-Word-Vectors/sgns.merge.char
/data0/nfs_data/zhaoxi9/pretrained_language_model/Chinese-Word-Vectors/sgns.merge.word
/data0/nfs_data/zhaoxi9/pretrained_language_model/Chinese-Word-Vectors/sgns.sogou.bigram
/data0/nfs_data/zhaoxi9/pretrained_language_model/Chinese-Word-Vectors/sgns.sogou.char
/data0/nfs_data/zhaoxi9/pretrained_language_model/Chinese-Word-Vectors/sgns.sogou.word
/data0/nfs_data/zhaoxi9/pretrained_language_model/Tencent_AILab_ChineseEmbedding.txt
/data0/nfs_data/zhaoxi9/pretrained_language_model/baike_26g_news_13g_novel_229g.bin



TextCNN1D
python -u run.py --ngrams_word 1 --min_freq_word 4 --batch_size 32 --learning_rate 2e-4 --dropout 0.5 --gpu 0 --model_name textcnn1d --embedding_file /data0/nfs_data/zhaoxi9/pretrained_language_model/Chinese-Word-Vectors/merge_sgns_bigram_char300.txt --padding_idx   dev acc=55.27%	test acc=55.37%
python -u run.py --ngrams_word 1 --min_freq_word 4 --batch_size 32 --learning_rate 2e-4 --dropout 0.5 --gpu 0 --model_name textcnn1d --embedding_file /data0/nfs_data/zhaoxi9/pretrained_language_model/Chinese-Word-Vectors/merge_sgns_bigram_char300.txt --padding_idx --freeze   dev acc=55.10%	test acc=54.74%

TextCNN2D
python -u run.py --ngrams_word 1 --min_freq_word 4 --batch_size 32 --learning_rate 2e-4 --dropout 0.5 --gpu 1 --model_name textcnn2d --embedding_file /data0/nfs_data/zhaoxi9/pretrained_language_model/Chinese-Word-Vectors/merge_sgns_bigram_char300.txt --padding_idx   dev acc=54.97%	test acc=55.70%  
***python -u run.py --ngrams_word 1 --min_freq_word 4 --batch_size 32 --learning_rate 2e-4 --dropout 0.5 --gpu 1 --model_name textcnn2d --embedding_file /data0/nfs_data/zhaoxi9/pretrained_language_model/Chinese-Word-Vectors/merge_sgns_bigram_char300.txt --padding_idx --freeze   dev acc=55.38%	test acc=55.22%***  

TextRNN
python -u run.py --ngrams_word 1 --min_freq_word 4 --batch_size 32 --learning_rate 2e-4 --dropout 0.5 --gpu 1 --model_name textrnn --embedding_file /data0/nfs_data/zhaoxi9/pretrained_language_model/Chinese-Word-Vectors/merge_sgns_bigram_char300.txt --padding_idx   dev acc=52.53%	test acc=53.10%
python -u run.py --ngrams_word 1 --min_freq_word 4 --batch_size 32 --learning_rate 2e-4 --dropout 0.5 --gpu 1 --model_name textrnn --embedding_file /data0/nfs_data/zhaoxi9/pretrained_language_model/Chinese-Word-Vectors/merge_sgns_bigram_char300.txt --padding_idx --freeze   dev acc=53.28%	test acc=53.92%

TextRNN_ATT
python -u run.py --ngrams_word 1 --min_freq_word 4 --batch_size 32 --learning_rate 2e-4 --dropout 0.5 --gpu 0 --model_name textrnn_att --embedding_file /data0/nfs_data/zhaoxi9/pretrained_language_model/Chinese-Word-Vectors/merge_sgns_bigram_char300.txt --padding_idx   dev acc=53.47%	test acc=54.43%
python -u run.py --ngrams_word 1 --min_freq_word 4 --batch_size 32 --learning_rate 2e-4 --dropout 0.5 --gpu 0 --model_name textrnn_att --embedding_file /data0/nfs_data/zhaoxi9/pretrained_language_model/Chinese-Word-Vectors/merge_sgns_bigram_char300.txt --padding_idx --freeze   dev acc=53.49%	test acc=53.25%

TextRCNN
python -u run.py --ngrams_word 1 --min_freq_word 4 --batch_size 32 --learning_rate 2e-4 --dropout 0.5 --gpu 1 --model_name textrcnn --embedding_file /data0/nfs_data/zhaoxi9/pretrained_language_model/Chinese-Word-Vectors/merge_sgns_bigram_char300.txt --padding_idx   dev acc=54.03%	test acc=54.40%
python -u run.py --ngrams_word 1 --min_freq_word 4 --batch_size 32 --learning_rate 2e-4 --dropout 0.5 --gpu 1 --model_name textrcnn --embedding_file /data0/nfs_data/zhaoxi9/pretrained_language_model/Chinese-Word-Vectors/merge_sgns_bigram_char300.txt --padding_idx --freeze   dev acc=53.34%	test acc=54.45%

DPCNN
python -u run.py --ngrams_word 1 --min_freq_word 4 --batch_size 32 --learning_rate 2e-4 --dropout 0.5 --gpu 0 --model_name dpcnn --embedding_file /data0/nfs_data/zhaoxi9/pretrained_language_model/Chinese-Word-Vectors/merge_sgns_bigram_char300.txt --padding_idx   dev acc=53.13%	test acc=53.83%
python -u run.py --ngrams_word 1 --min_freq_word 4 --batch_size 32 --learning_rate 2e-4 --dropout 0.5 --gpu 0 --model_name dpcnn --embedding_file /data0/nfs_data/zhaoxi9/pretrained_language_model/Chinese-Word-Vectors/merge_sgns_bigram_char300.txt --padding_idx --freeze   dev acc=53.24%	test acc=53.64%

Transformer
python -u run.py --ngrams_word 1 --min_freq_word 4 --batch_size 32 --learning_rate 2e-4 --dropout 0.5 --gpu 0 --model_name transformer --embedding_file /data0/nfs_data/zhaoxi9/pretrained_language_model/Chinese-Word-Vectors/merge_sgns_bigram_char300.txt --padding_idx   dev acc=53.34%	test acc=53.32%
python -u run.py --ngrams_word 1 --min_freq_word 4 --batch_size 32 --learning_rate 2e-4 --dropout 0.5 --gpu 0 --model_name transformer --embedding_file /data0/nfs_data/zhaoxi9/pretrained_language_model/Chinese-Word-Vectors/merge_sgns_bigram_char300.txt --padding_idx --freeze   dev acc=52.79%	test acc=52.98%



Pretrained Language Model

nohup python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/bert-base-chinese --gpu 1 > nohup_bert-base-chinese.out 2>&1 &     dev acc=9.75%  test acc=9.56%
nohup python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/bert-base-chinese --gpu 2 --freeze > nohup_bert-base-chinese_freeze.out 2>&1 &     dev acc=49.29%  test acc=48.51%

nohup python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/ckiplab_albert-tiny-chinese --gpu 1 > nohup_ckiplab_albert-tiny-chinese.out 2>&1 &     dev acc=%  test acc=%
nohup python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/ckiplab_albert-tiny-chinese --gpu 1 --freeze > nohup_ckiplab_albert-tiny-chinese_freeze.out 2>&1 &     dev acc=%  test acc=%
nohup python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/ckiplab_bert-base-chinese --gpu 1 > nohup_ckiplab_bert-base-chinese.out 2>&1 &     dev acc=%  test acc=%
nohup python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/ckiplab_bert-base-chinese --gpu 1 --freeze > nohup_ckiplab_bert-base-chinese_freeze.out 2>&1 &     dev acc=%  test acc=%

nohup python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/clue_albert_chinese_small --gpu 1 > nohup_clue_albert_chinese_small.out 2>&1 &     dev acc=%  test acc=%
nohup python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/clue_albert_chinese_small --gpu 1 --freeze > nohup_clue_albert_chinese_small_freeze.out 2>&1 &     dev acc=%  test acc=%
nohup python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/clue_albert_chinese_tiny --gpu 1 > nohup_clue_albert_chinese_tiny.out 2>&1 &     dev acc=43.72%  test acc=43.62%
nohup python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/clue_albert_chinese_tiny --gpu 1 --freeze > nohup_clue_albert_chinese_tiny_freeze.out 2>&1 &     dev acc=40.76%  test acc=40.28%
nohup python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/clue_roberta_chinese_3L312_clue_tiny --gpu 1 > nohup_clue_roberta_chinese_3L312_clue_tiny.out 2>&1 &     dev acc=%  test acc=%
nohup python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/clue_roberta_chinese_3L312_clue_tiny --gpu 1 --freeze > nohup_clue_roberta_chinese_3L312_clue_tiny_freeze.out 2>&1 &     dev acc=%  test acc=%
nohup python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/clue_roberta_chinese_3L768_clue_tiny --gpu 1 > nohup_clue_roberta_chinese_3L768_clue_tiny.out 2>&1 &     dev acc=%  test acc=%
nohup python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/clue_roberta_chinese_3L768_clue_tiny --gpu 1 --freeze > nohup_clue_roberta_chinese_3L768_clue_tiny_freeze.out 2>&1 &     dev acc=%  test acc=%
nohup python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/clue_roberta_chinese_base --gpu 1 > nohup_clue_roberta_chinese_base.out 2>&1 &     dev acc=%  test acc=%
nohup python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/clue_roberta_chinese_base --gpu 1 --freeze > nohup_clue_roberta_chinese_base_freeze.out 2>&1 &     dev acc=%  test acc=%
nohup python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/clue_roberta_chinese_clue_large --gpu 1 > nohup_clue_roberta_chinese_clue_large.out 2>&1 &     dev acc=%  test acc=%
nohup python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/clue_roberta_chinese_clue_large --gpu 1 --freeze > nohup_clue_roberta_chinese_clue_large_freeze.out 2>&1 &     dev acc=%  test acc=%
nohup python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/clue_roberta_chinese_clue_tiny --gpu 1 > nohup_clue_roberta_chinese_clue_tiny.out 2>&1 &     dev acc=%  test acc=%
nohup python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/clue_roberta_chinese_clue_tiny --gpu 1 --freeze > nohup_clue_roberta_chinese_clue_tiny_freeze.out 2>&1 &     dev acc=%  test acc=%
nohup python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/clue_roberta_chinese_large --gpu 1 > nohup_clue_roberta_chinese_large.out 2>&1 &     dev acc=%  test acc=%
nohup python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/clue_roberta_chinese_large --gpu 1 --freeze > nohup_clue_roberta_chinese_large_freeze.out 2>&1 &     dev acc=%  test acc=%
nohup python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/clue_roberta_chinese_pair_large --gpu 1 > nohup_clue_roberta_chinese_pair_large.out 2>&1 &     dev acc=%  test acc=%
nohup python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/clue_roberta_chinese_pair_large --gpu 1 --freeze > nohup_clue_roberta_chinese_pair_large_freeze.out 2>&1 &     dev acc=%  test acc=%
nohup python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/clue_roberta_chinese_pair_tiny --gpu 1 > nohup_clue_roberta_chinese_pair_tiny.out 2>&1 &     dev acc=%  test acc=%
nohup python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/clue_roberta_chinese_pair_tiny --gpu 1 --freeze > nohup_clue_roberta_chinese_pair_tiny_freeze.out 2>&1 &     dev acc=%  test acc=%
nohup python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/clue_xlnet_chinese_large --gpu 1 > nohup_clue_xlnet_chinese_large.out 2>&1 &     dev acc=%  test acc=%
nohup python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/clue_xlnet_chinese_large --gpu 1 --freeze > nohup_clue_xlnet_chinese_large_freeze.out 2>&1 &     dev acc=%  test acc=%

nohup python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/hfl_chinese-bert-wwm --gpu 1 > nohup_hfl_chinese-bert-wwm.out 2>&1 &     dev acc=%  test acc=%
nohup python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/hfl_chinese-bert-wwm --gpu 1 --freeze > nohup_hfl_chinese-bert-wwm_freeze.out 2>&1 &     dev acc=%  test acc=%
nohup python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/hfl_chinese-bert-wwm-ext --gpu 1 > nohup_hfl_chinese-bert-wwm-ext.out 2>&1 &     dev acc=%  test acc=%
nohup python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/hfl_chinese-bert-wwm-ext --gpu 1 --freeze > nohup_hfl_chinese-bert-wwm-ext_freeze.out 2>&1 &     dev acc=%  test acc=%
nohup python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/hfl_chinese-electra-180g-base-discriminator --gpu 1 > nohup_hfl_chinese-electra-180g-base-discriminator.out 2>&1 &     dev acc=%  test acc=%
nohup python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/hfl_chinese-electra-180g-base-discriminator --gpu 1 --freeze > nohup_hfl_chinese-electra-180g-base-discriminator_freeze.out 2>&1 &     dev acc=%  test acc=%
nohup python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/hfl_chinese-electra-180g-large-discriminator --gpu 1 > nohup_hfl_chinese-electra-180g-large-discriminator.out 2>&1 &     dev acc=%  test acc=%
nohup python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/hfl_chinese-electra-180g-large-discriminator --gpu 1 --freeze > nohup_hfl_chinese-electra-180g-large-discriminator_freeze.out 2>&1 &     dev acc=%  test acc=%
nohup python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/hfl_chinese-electra-180g-small-discriminator --gpu 1 > nohup_hfl_chinese-electra-180g-small-discriminator.out 2>&1 &     dev acc=%  test acc=%
nohup python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/hfl_chinese-electra-180g-small-discriminator --gpu 1 --freeze > nohup_hfl_chinese-electra-180g-small-discriminator_freeze.out 2>&1 &     dev acc=%  test acc=%
nohup python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/hfl_chinese-electra-180g-small-ex-discriminator --gpu 1 > nohup_hfl_chinese-electra-180g-small-ex-discriminator.out 2>&1 &     dev acc=%  test acc=%
nohup python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/hfl_chinese-electra-180g-small-ex-discriminator --gpu 1 --freeze > nohup_hfl_chinese-electra-180g-small-ex-discriminator_freeze.out 2>&1 &     dev acc=%  test acc=%
nohup python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/hfl_chinese-electra-base-discriminator --gpu 1 > nohup_hfl_chinese-electra-base-discriminator.out 2>&1 &     dev acc=%  test acc=%
nohup python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/hfl_chinese-electra-base-discriminator --gpu 1 --freeze > nohup_hfl_chinese-electra-base-discriminator_freeze.out 2>&1 &     dev acc=%  test acc=%
nohup python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/hfl_chinese-electra-base-generator --gpu 1 > nohup_hfl_chinese-electra-base-generator.out 2>&1 &     dev acc=%  test acc=%
nohup python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/hfl_chinese-electra-base-generator --gpu 1 --freeze > nohup_hfl_chinese-electra-base-generator_freeze.out 2>&1 &     dev acc=%  test acc=%
nohup python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/hfl_chinese-electra-large-discriminator --gpu 1 > nohup_hfl_chinese-electra-large-discriminator.out 2>&1 &     dev acc=%  test acc=%
nohup python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/hfl_chinese-electra-large-discriminator --gpu 1 --freeze > nohup_hfl_chinese-electra-large-discriminator_freeze.out 2>&1 &     dev acc=%  test acc=%
nohup python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/hfl_chinese-electra-large-generator --gpu 1 > nohup_hfl_chinese-electra-large-generator.out 2>&1 &     dev acc=%  test acc=%
nohup python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/hfl_chinese-electra-large-generator --gpu 1 --freeze > nohup_hfl_chinese-electra-large-generator_freeze.out 2>&1 &     dev acc=%  test acc=%
nohup python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/hfl_chinese-electra-small-ex-discriminator --gpu 1 > nohup_hfl_chinese-electra-small-ex-discriminator.out 2>&1 &     dev acc=%  test acc=%
nohup python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/hfl_chinese-electra-small-ex-discriminator --gpu 1 --freeze > nohup_hfl_chinese-electra-small-ex-discriminator_freeze.out 2>&1 &     dev acc=%  test acc=%
nohup python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/hfl_chinese-electra-small-ex-generator --gpu 1 > nohup_hfl_chinese-electra-small-ex-generator.out 2>&1 &     dev acc=%  test acc=%
nohup python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/hfl_chinese-electra-small-ex-generator --gpu 1 --freeze > nohup_hfl_chinese-electra-small-ex-generator_freeze.out 2>&1 &     dev acc=%  test acc=%
nohup python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/hfl_chinese-legal-electra-base-generator --gpu 1 > nohup_hfl_chinese-legal-electra-base-generator.out 2>&1 &     dev acc=%  test acc=%
nohup python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/hfl_chinese-legal-electra-base-generator --gpu 1 --freeze > nohup_hfl_chinese-legal-electra-base-generator_freeze.out 2>&1 &     dev acc=%  test acc=%
nohup python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/hfl_chinese-legal-electra-large-generator --gpu 1 > nohup_hfl_chinese-legal-electra-large-generator.out 2>&1 &     dev acc=%  test acc=%
nohup python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/hfl_chinese-legal-electra-large-generator --gpu 1 --freeze > nohup_hfl_chinese-legal-electra-large-generator_freeze.out 2>&1 &     dev acc=%  test acc=%
nohup python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/hfl_chinese-legal-electra-small-discriminator --gpu 1 > nohup_hfl_chinese-legal-electra-small-discriminator.out 2>&1 &     dev acc=%  test acc=%
nohup python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/hfl_chinese-legal-electra-small-discriminator --gpu 1 --freeze > nohup_hfl_chinese-legal-electra-small-discriminator_freeze.out 2>&1 &     dev acc=%  test acc=%
nohup python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/hfl_chinese-legal-electra-small-generator --gpu 1 > nohup_hfl_chinese-legal-electra-small-generator.out 2>&1 &     dev acc=%  test acc=%
nohup python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/hfl_chinese-legal-electra-small-generator --gpu 1 --freeze > nohup_hfl_chinese-legal-electra-small-generator_freeze.out 2>&1 &     dev acc=%  test acc=%
nohup python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/hfl_chinese-macbert-base --gpu 1 > nohup_hfl_chinese-macbert-base.out 2>&1 &     dev acc=%  test acc=%
nohup python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/hfl_chinese-macbert-base --gpu 1 --freeze > nohup_hfl_chinese-macbert-base_freeze.out 2>&1 &     dev acc=%  test acc=%
nohup python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/hfl_chinese-macbert-large --gpu 1 > nohup_hfl_chinese-macbert-large.out 2>&1 &     dev acc=%  test acc=%
nohup python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/hfl_chinese-macbert-large --gpu 1 --freeze > nohup_hfl_chinese-macbert-large_freeze.out 2>&1 &     dev acc=%  test acc=%
nohup python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/hfl_chinese-pert-base --gpu 1 > nohup_hfl_chinese-pert-base.out 2>&1 &     dev acc=%  test acc=%
nohup python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/hfl_chinese-pert-base --gpu 1 --freeze > nohup_hfl_chinese-pert-base_freeze.out 2>&1 &     dev acc=%  test acc=%
nohup python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/hfl_chinese-pert-large --gpu 1 > nohup_hfl_chinese-pert-large.out 2>&1 &     dev acc=%  test acc=%
nohup python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/hfl_chinese-pert-large --gpu 1 --freeze > nohup_hfl_chinese-pert-large_freeze.out 2>&1 &     dev acc=%  test acc=%
nohup python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/hfl_chinese-roberta-wwm-ext --gpu 0 > nohup_hfl_chinese-roberta-wwm-ext.out 2>&1 &     dev acc=%  test acc=%
nohup python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/hfl_chinese-roberta-wwm-ext --gpu 1 --freeze > nohup_hfl_chinese-roberta-wwm-ext_freeze.out 2>&1 &     dev acc=54.57%  test acc=54.15%
nohup python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/hfl_chinese-roberta-wwm-ext-large --gpu 2 > nohup_hfl_chinese-roberta-wwm-ext-large.out 2>&1 &     dev acc=%  test acc=%
nohup python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/hfl_chinese-roberta-wwm-ext-large --gpu 3 --freeze > nohup_hfl_chinese-roberta-wwm-ext-large_freeze.out 2>&1 &     dev acc=56.48%  test acc=56.09%
nohup python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/hfl_chinese-xlnet-base --gpu 1 > nohup_hfl_chinese-xlnet-base.out 2>&1 &     dev acc=%  test acc=%
nohup python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/hfl_chinese-xlnet-base --gpu 1 --freeze > nohup_hfl_chinese-xlnet-base_freeze.out 2>&1 &     dev acc=%  test acc=%
nohup python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/hfl_chinese-xlnet-mid --gpu 1 > nohup_hfl_chinese-xlnet-mid.out 2>&1 &     dev acc=%  test acc=%
nohup python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/hfl_chinese-xlnet-mid --gpu 1 --freeze > nohup_hfl_chinese-xlnet-mid_freeze.out 2>&1 &     dev acc=%  test acc=%

nohup python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/hfl_rbt3 --gpu 1 > nohup_hfl_rbt3.out 2>&1 &     dev acc=%  test acc=%
nohup python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/hfl_rbt3 --gpu 1 --freeze > nohup_hfl_rbt3_freeze.out 2>&1 &     dev acc=%  test acc=%
nohup python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/hfl_rbt4 --gpu 1 > nohup_hfl_rbt4.out 2>&1 &     dev acc=%  test acc=%
nohup python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/hfl_rbt4 --gpu 1 --freeze > nohup_hfl_rbt4_freeze.out 2>&1 &     dev acc=%  test acc=%
nohup python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/hfl_rbt6 --gpu 1 > nohup_hfl_rbt6.out 2>&1 &     dev acc=%  test acc=%
nohup python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/hfl_rbt6 --gpu 1 --freeze > nohup_hfl_rbt6_freeze.out 2>&1 &     dev acc=%  test acc=%
nohup python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/hfl_rbtl3 --gpu 1 > nohup_hfl_rbtl3.out 2>&1 &     dev acc=%  test acc=%
nohup python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/hfl_rbtl3 --gpu 1 --freeze > nohup_hfl_rbtl3_freeze.out 2>&1 &     dev acc=%  test acc=%

nohup python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/junnyu_roformer_chinese_sim_char_ft_small --gpu 1 > nohup_junnyu_roformer_chinese_sim_char_ft_small.out 2>&1 &     dev acc=%  test acc=%
nohup python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/junnyu_roformer_chinese_sim_char_ft_small --gpu 1 --freeze > nohup_junnyu_roformer_chinese_sim_char_ft_small_freeze.out 2>&1 &     dev acc=%  test acc=%

nohup python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/luhua_chinese_pretrain_mrc_macbert_large --gpu 1 > nohup_luhua_chinese_pretrain_mrc_macbert_large.out 2>&1 &     dev acc=%  test acc=%
nohup python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/luhua_chinese_pretrain_mrc_macbert_large --gpu 1 --freeze > nohup_luhua_chinese_pretrain_mrc_macbert_large_freeze.out 2>&1 &     dev acc=%  test acc=%
nohup python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/luhua_chinese_pretrain_mrc_roberta_wwm_ext_large --gpu 1 > nohup_luhua_chinese_pretrain_mrc_roberta_wwm_ext_large.out 2>&1 &     dev acc=%  test acc=%
nohup python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/luhua_chinese_pretrain_mrc_roberta_wwm_ext_large --gpu 1 --freeze > nohup_luhua_chinese_pretrain_mrc_roberta_wwm_ext_large_freeze.out 2>&1 &     dev acc=%  test acc=%

nohup python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/peterchou_nezha-chinese-base --gpu 1 > nohup_peterchou_nezha-chinese-base.out 2>&1 &     dev acc=%  test acc=%
nohup python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/peterchou_nezha-chinese-base --gpu 1 --freeze > nohup_peterchou_nezha-chinese-base_freeze.out 2>&1 &     dev acc=%  test acc=%

nohup python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/uer_chinese_roberta_L-4_H-512 --gpu 1 > nohup_uer_chinese_roberta_L-4_H-512.out 2>&1 &     dev acc=%  test acc=%
nohup python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/uer_chinese_roberta_L-4_H-512 --gpu 1 --freeze > nohup_uer_chinese_roberta_L-4_H-512_freeze.out 2>&1 &     dev acc=%  test acc=%
nohup python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/uer_roberta-base-finetuned-jd-full-chinese --gpu 1 > nohup_uer_roberta-base-finetuned-jd-full-chinese.out 2>&1 &     dev acc=%  test acc=%
nohup python -u run_fine_tuning_huggingface_models.py --model_name /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/uer_roberta-base-finetuned-jd-full-chinese --gpu 1 --freeze > nohup_uer_roberta-base-finetuned-jd-full-chinese_freeze.out 2>&1 &     dev acc=%  test acc=%
