#encoding=utf-8

# import torch
# from transformers import BertTokenizer, AlbertModel
# from transformers import AutoConfig

# tokenizer = BertTokenizer.from_pretrained('/Users/chang/workspace_of_python/pretrained_language_model/huggingface_pretrained_models/clue_albert_chinese_tiny/')
# model = AlbertModel.from_pretrained("/Users/chang/workspace_of_python/pretrained_language_model/huggingface_pretrained_models/clue_albert_chinese_tiny/")
# print('ok')


# # # placed before import transformers
# # import os
# # # transformers environment
# # os.environ['TRANSFORMERS_CACHE'] = './tmp/'
# # # torch environment
# # os.environ['TORCH_HOME']='./tmp_a/'
# # import torch
# from transformers import BertTokenizer, AlbertModel
# tokenizer = BertTokenizer.from_pretrained("clue/albert_chinese_tiny")
# tokenizer = BertTokenizer.from_pretrained("clue/albert_chinese_tiny", cache_dir='./tmp/')
# print('ok')










# # bert-base-chinese
# from transformers import AutoTokenizer, AutoModel, BertTokenizer, BertModel
# model_name = '/data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/bert-base-chinese'
# tokenizer = BertTokenizer.from_pretrained(model_name)       # ok
# tokenizer = AutoTokenizer.from_pretrained(model_name)       # ok
# model = AutoModel.from_pretrained(model_name)               # ok
# model = BertModel.from_pretrained(model_name)               # ok
# print('end')






# hfl
# from transformers import AutoTokenizer, AutoModel, BertTokenizer, BertModel
# model_name = '/data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/hfl_chinese-bert-wwm'
# tokenizer = BertTokenizer.from_pretrained(model_name)       # ok
# tokenizer = AutoTokenizer.from_pretrained(model_name)       # ok
# model = AutoModel.from_pretrained(model_name)               # ok
# model = BertModel.from_pretrained(model_name)               # ok
# print('end')
# from transformers import AutoTokenizer, AutoModel, BertTokenizer, BertModel
# model_name = '/data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/hfl_chinese-bert-wwm-ext'
# tokenizer = BertTokenizer.from_pretrained(model_name)       # ok
# tokenizer = AutoTokenizer.from_pretrained(model_name)       # ok
# model = AutoModel.from_pretrained(model_name)               # ok
# model = BertModel.from_pretrained(model_name)               # ok
# print('end')
# from transformers import AutoTokenizer, AutoModel, BertTokenizer, BertModel
# model_name = '/data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/hfl_chinese-roberta-wwm-ext'
# tokenizer = BertTokenizer.from_pretrained(model_name)       # ok
# tokenizer = AutoTokenizer.from_pretrained(model_name)       # ok
# model = AutoModel.from_pretrained(model_name)               # ok
# model = BertModel.from_pretrained(model_name)               # ok
# print('end')
# from transformers import AutoTokenizer, AutoModel, BertTokenizer, BertModel
# model_name = '/data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/hfl_chinese-roberta-wwm-ext-large'
# tokenizer = BertTokenizer.from_pretrained(model_name)       # ok
# tokenizer = AutoTokenizer.from_pretrained(model_name)       # ok
# model = AutoModel.from_pretrained(model_name)               # ok
# model = BertModel.from_pretrained(model_name)               # ok
# print('end')
# from transformers import AutoTokenizer, AutoModel, BertTokenizer, BertModel
# model_name = '/data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/hfl_rbt3'
# tokenizer = BertTokenizer.from_pretrained(model_name)       # ok
# tokenizer = AutoTokenizer.from_pretrained(model_name)       # ok
# model = AutoModel.from_pretrained(model_name)               # ok
# model = BertModel.from_pretrained(model_name)               # ok
# print('end')
# from transformers import AutoTokenizer, AutoModel, BertTokenizer, BertModel
# model_name = '/data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/hfl_rbt4'
# tokenizer = BertTokenizer.from_pretrained(model_name)       # ok
# tokenizer = AutoTokenizer.from_pretrained(model_name)       # ok
# model = AutoModel.from_pretrained(model_name)               # ok
# model = BertModel.from_pretrained(model_name)               # ok
# print('end')
# from transformers import AutoTokenizer, AutoModel, BertTokenizer, BertModel
# model_name = '/data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/hfl_rbt6'
# tokenizer = BertTokenizer.from_pretrained(model_name)       # ok
# tokenizer = AutoTokenizer.from_pretrained(model_name)       # ok
# model = AutoModel.from_pretrained(model_name)               # ok
# model = BertModel.from_pretrained(model_name)               # ok
# print('end')
# from transformers import AutoTokenizer, AutoModel, BertTokenizer, BertModel
# model_name = '/data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/hfl_rbtl3'
# tokenizer = BertTokenizer.from_pretrained(model_name)       # ok
# tokenizer = AutoTokenizer.from_pretrained(model_name)       # ok
# model = AutoModel.from_pretrained(model_name)               # ok
# model = BertModel.from_pretrained(model_name)               # ok
# print('end')

# from transformers import AutoTokenizer, AutoModel, ElectraTokenizer, ElectraModel
# model_name = '/data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/hfl_chinese-electra-180g-base-discriminator'
# tokenizer = ElectraTokenizer.from_pretrained(model_name)       # ok
# tokenizer = AutoTokenizer.from_pretrained(model_name)       # ok
# model = AutoModel.from_pretrained(model_name)               # ok
# model = ElectraModel.from_pretrained(model_name)               # ok
# print('end')
# from transformers import AutoTokenizer, AutoModel, ElectraTokenizer, ElectraModel
# model_name = '/data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/hfl_chinese-electra-180g-large-discriminator'
# tokenizer = ElectraTokenizer.from_pretrained(model_name)       # ok
# tokenizer = AutoTokenizer.from_pretrained(model_name)       # ok
# model = AutoModel.from_pretrained(model_name)               # ok
# model = ElectraModel.from_pretrained(model_name)               # ok
# print('end')
# from transformers import AutoTokenizer, AutoModel, ElectraTokenizer, ElectraModel
# model_name = '/data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/hfl_chinese-electra-180g-small-discriminator'
# tokenizer = ElectraTokenizer.from_pretrained(model_name)       # ok
# tokenizer = AutoTokenizer.from_pretrained(model_name)       # ok
# model = AutoModel.from_pretrained(model_name)               # ok
# model = ElectraModel.from_pretrained(model_name)               # ok
# print('end')
# from transformers import AutoTokenizer, AutoModel, ElectraTokenizer, ElectraModel
# model_name = '/data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/hfl_chinese-electra-180g-small-ex-discriminator'
# tokenizer = ElectraTokenizer.from_pretrained(model_name)       # ok
# tokenizer = AutoTokenizer.from_pretrained(model_name)       # ok
# model = AutoModel.from_pretrained(model_name)               # ok
# model = ElectraModel.from_pretrained(model_name)               # ok
# print('end')
# from transformers import AutoTokenizer, AutoModel, ElectraTokenizer, ElectraModel
# model_name = '/data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/hfl_chinese-electra-base-discriminator'
# tokenizer = ElectraTokenizer.from_pretrained(model_name)       # ok
# tokenizer = AutoTokenizer.from_pretrained(model_name)       # ok
# model = AutoModel.from_pretrained(model_name)               # ok
# model = ElectraModel.from_pretrained(model_name)               # ok
# print('end')
# from transformers import AutoTokenizer, AutoModel, ElectraTokenizer, ElectraModel
# model_name = '/data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/hfl_chinese-electra-base-generator'
# tokenizer = ElectraTokenizer.from_pretrained(model_name)       # ok
# tokenizer = AutoTokenizer.from_pretrained(model_name)       # ok
# model = AutoModel.from_pretrained(model_name)               # ok
# model = ElectraModel.from_pretrained(model_name)               # ok
# print('end')
# from transformers import AutoTokenizer, AutoModel, ElectraTokenizer, ElectraModel
# model_name = '/data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/hfl_chinese-electra-large-discriminator'
# tokenizer = ElectraTokenizer.from_pretrained(model_name)       # ok
# tokenizer = AutoTokenizer.from_pretrained(model_name)       # ok
# model = AutoModel.from_pretrained(model_name)               # ok
# model = ElectraModel.from_pretrained(model_name)               # ok
# print('end')
# from transformers import AutoTokenizer, AutoModel, ElectraTokenizer, ElectraModel
# model_name = '/data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/hfl_chinese-electra-large-generator'
# tokenizer = ElectraTokenizer.from_pretrained(model_name)       # ok
# tokenizer = AutoTokenizer.from_pretrained(model_name)       # ok
# model = AutoModel.from_pretrained(model_name)               # ok
# model = ElectraModel.from_pretrained(model_name)               # ok
# print('end')
# from transformers import AutoTokenizer, AutoModel, ElectraTokenizer, ElectraModel
# model_name = '/data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/hfl_chinese-electra-small-ex-discriminator'
# tokenizer = ElectraTokenizer.from_pretrained(model_name)       # ok
# tokenizer = AutoTokenizer.from_pretrained(model_name)       # ok
# model = AutoModel.from_pretrained(model_name)               # ok
# model = ElectraModel.from_pretrained(model_name)               # ok
# print('end')
# from transformers import AutoTokenizer, AutoModel, ElectraTokenizer, ElectraModel
# model_name = '/data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/hfl_chinese-electra-small-ex-generator'
# tokenizer = ElectraTokenizer.from_pretrained(model_name)       # ok
# tokenizer = AutoTokenizer.from_pretrained(model_name)       # ok
# model = AutoModel.from_pretrained(model_name)               # ok
# model = ElectraModel.from_pretrained(model_name)               # ok
# print('end')
# from transformers import AutoTokenizer, AutoModel, ElectraTokenizer, ElectraModel
# model_name = '/data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/hfl_chinese-legal-electra-base-generator'
# tokenizer = ElectraTokenizer.from_pretrained(model_name)       # ok
# tokenizer = AutoTokenizer.from_pretrained(model_name)       # ok
# model = AutoModel.from_pretrained(model_name)               # ok
# model = ElectraModel.from_pretrained(model_name)               # ok
# print('end')
# from transformers import AutoTokenizer, AutoModel, ElectraTokenizer, ElectraModel
# model_name = '/data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/hfl_chinese-legal-electra-large-generator'
# tokenizer = ElectraTokenizer.from_pretrained(model_name)       # ok
# tokenizer = AutoTokenizer.from_pretrained(model_name)       # ok
# model = AutoModel.from_pretrained(model_name)               # ok
# model = ElectraModel.from_pretrained(model_name)               # ok
# print('end')
# from transformers import AutoTokenizer, AutoModel, ElectraTokenizer, ElectraModel
# model_name = '/data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/hfl_chinese-legal-electra-small-discriminator'
# tokenizer = ElectraTokenizer.from_pretrained(model_name)       # ok
# tokenizer = AutoTokenizer.from_pretrained(model_name)       # ok
# model = AutoModel.from_pretrained(model_name)               # ok
# model = ElectraModel.from_pretrained(model_name)               # ok
# print('end')
# from transformers import AutoTokenizer, AutoModel, ElectraTokenizer, ElectraModel
# model_name = '/data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/hfl_chinese-legal-electra-small-generator'
# tokenizer = ElectraTokenizer.from_pretrained(model_name)       # ok
# tokenizer = AutoTokenizer.from_pretrained(model_name)       # ok
# model = AutoModel.from_pretrained(model_name)               # ok
# model = ElectraModel.from_pretrained(model_name)               # ok
# print('end')

# from transformers import AutoTokenizer, AutoModel, XLNetTokenizer, XLNetModel
# model_name = '/data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/hfl_chinese-xlnet-base'
# tokenizer = XLNetTokenizer.from_pretrained(model_name)       # ok
# tokenizer = AutoTokenizer.from_pretrained(model_name)       # ok
# model = AutoModel.from_pretrained(model_name)               # ok
# model = XLNetModel.from_pretrained(model_name)               # ok
# print('end')
# from transformers import AutoTokenizer, AutoModel, XLNetTokenizer, XLNetModel
# model_name = '/data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/hfl_chinese-xlnet-mid'
# tokenizer = XLNetTokenizer.from_pretrained(model_name)       # ok
# tokenizer = AutoTokenizer.from_pretrained(model_name)       # ok
# model = AutoModel.from_pretrained(model_name)               # ok
# model = XLNetModel.from_pretrained(model_name)               # ok
# print('end')

# from transformers import AutoTokenizer, AutoModel, BertTokenizer, BertModel
# model_name = '/data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/hfl_chinese-macbert-base'
# tokenizer = BertTokenizer.from_pretrained(model_name)       # ok
# tokenizer = AutoTokenizer.from_pretrained(model_name)       # ok
# model = AutoModel.from_pretrained(model_name)               # ok
# model = BertModel.from_pretrained(model_name)               # ok
# print('end')
# from transformers import AutoTokenizer, AutoModel, BertTokenizer, BertModel
# model_name = '/data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/hfl_chinese-macbert-large'
# tokenizer = BertTokenizer.from_pretrained(model_name)       # ok
# tokenizer = AutoTokenizer.from_pretrained(model_name)       # ok
# model = AutoModel.from_pretrained(model_name)               # ok
# model = BertModel.from_pretrained(model_name)               # ok
# print('end')

# from transformers import AutoTokenizer, AutoModel, BertTokenizer, BertModel
# model_name = '/data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/hfl_chinese-pert-base'
# tokenizer = BertTokenizer.from_pretrained(model_name)       # ok
# tokenizer = AutoTokenizer.from_pretrained(model_name)       # ok
# model = AutoModel.from_pretrained(model_name)               # ok
# model = BertModel.from_pretrained(model_name)               # ok
# print('end')
# from transformers import AutoTokenizer, AutoModel, BertTokenizer, BertModel
# model_name = '/data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/hfl_chinese-pert-large'
# tokenizer = BertTokenizer.from_pretrained(model_name)       # ok
# tokenizer = AutoTokenizer.from_pretrained(model_name)       # ok
# model = AutoModel.from_pretrained(model_name)               # ok
# model = BertModel.from_pretrained(model_name)               # ok
# print('end')




# # clue
# # clue_albert_chinese_tiny
# from transformers import AutoTokenizer, AutoModel, BertTokenizer, AlbertModel
# model_name = '/data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/clue_albert_chinese_small'
# tokenizer = BertTokenizer.from_pretrained(model_name)       # ok
# # tokenizer = AutoTokenizer.from_pretrained(model_name)     # error
# model = AutoModel.from_pretrained(model_name)               # ok
# model = AlbertModel.from_pretrained(model_name)               # ok
# print('end')
# from transformers import AutoTokenizer, AutoModel, BertTokenizer, AlbertModel
# model_name = '/data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/clue_albert_chinese_tiny'
# tokenizer = BertTokenizer.from_pretrained(model_name)       # ok
# # tokenizer = AutoTokenizer.from_pretrained(model_name)     # error
# model = AutoModel.from_pretrained(model_name)               # ok
# model = AlbertModel.from_pretrained(model_name)               # ok
# print('end')

# from transformers import AutoTokenizer, AutoModel, BertTokenizer, BertModel
# model_name = '/data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/clue_roberta_chinese_3L312_clue_tiny'
# tokenizer = BertTokenizer.from_pretrained(model_name)       # ok
# # tokenizer = AutoTokenizer.from_pretrained(model_name)     # error
# model = AutoModel.from_pretrained(model_name)               # ok
# model = BertModel.from_pretrained(model_name)               # ok
# print('end')
# from transformers import AutoTokenizer, AutoModel, BertTokenizer, BertModel
# model_name = '/data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/clue_roberta_chinese_3L768_clue_tiny'
# tokenizer = BertTokenizer.from_pretrained(model_name)       # ok
# # tokenizer = AutoTokenizer.from_pretrained(model_name)     # error
# model = AutoModel.from_pretrained(model_name)               # ok
# model = BertModel.from_pretrained(model_name)               # ok
# print('end')
# from transformers import AutoTokenizer, AutoModel, BertTokenizer, BertModel
# model_name = '/data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/clue_roberta_chinese_base'
# tokenizer = BertTokenizer.from_pretrained(model_name)       # ok
# # tokenizer = AutoTokenizer.from_pretrained(model_name)     # error
# model = AutoModel.from_pretrained(model_name)               # ok
# model = BertModel.from_pretrained(model_name)               # ok
# print('end')
# from transformers import AutoTokenizer, AutoModel, BertTokenizer, BertModel
# model_name = '/data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/clue_roberta_chinese_clue_large'
# tokenizer = BertTokenizer.from_pretrained(model_name)       # ok
# # tokenizer = AutoTokenizer.from_pretrained(model_name)     # error
# model = AutoModel.from_pretrained(model_name)               # ok
# model = BertModel.from_pretrained(model_name)               # ok
# print('end')
# from transformers import AutoTokenizer, AutoModel, BertTokenizer, BertModel
# model_name = '/data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/clue_roberta_chinese_clue_tiny'
# tokenizer = BertTokenizer.from_pretrained(model_name)       # ok
# # tokenizer = AutoTokenizer.from_pretrained(model_name)     # error
# model = AutoModel.from_pretrained(model_name)               # ok
# model = BertModel.from_pretrained(model_name)               # ok
# print('end')
# from transformers import AutoTokenizer, AutoModel, BertTokenizer, BertModel
# model_name = '/data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/clue_roberta_chinese_large'
# tokenizer = BertTokenizer.from_pretrained(model_name)       # ok
# # tokenizer = AutoTokenizer.from_pretrained(model_name)     # error
# model = AutoModel.from_pretrained(model_name)               # ok
# model = BertModel.from_pretrained(model_name)               # ok
# print('end')
# from transformers import AutoTokenizer, AutoModel, BertTokenizer, BertModel
# model_name = '/data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/clue_roberta_chinese_pair_large'
# tokenizer = BertTokenizer.from_pretrained(model_name)       # ok
# # tokenizer = AutoTokenizer.from_pretrained(model_name)     # error
# model = AutoModel.from_pretrained(model_name)               # ok
# model = BertModel.from_pretrained(model_name)               # ok
# print('end')
# from transformers import AutoTokenizer, AutoModel, BertTokenizer, BertModel
# model_name = '/data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/clue_roberta_chinese_pair_tiny'
# tokenizer = BertTokenizer.from_pretrained(model_name)       # ok
# # tokenizer = AutoTokenizer.from_pretrained(model_name)     # error
# model = AutoModel.from_pretrained(model_name)               # ok
# model = BertModel.from_pretrained(model_name)               # ok
# print('end')

# from transformers import AutoTokenizer, AutoModel, XLNetTokenizer, XLNetModel
# model_name = '/data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/clue_xlnet_chinese_large'
# tokenizer = XLNetTokenizer.from_pretrained(model_name)       # ok
# tokenizer = AutoTokenizer.from_pretrained(model_name)     # error
# model = AutoModel.from_pretrained(model_name)               # ok
# model = XLNetModel.from_pretrained(model_name)               # ok
# print('end')



# # ckiplab_albert-tiny-chinese
# from transformers import AutoTokenizer, AutoModel, BertTokenizer, AlbertModel
# model_name = '/data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/ckiplab_albert-tiny-chinese'
# tokenizer = BertTokenizer.from_pretrained(model_name)       # ok
# tokenizer = AutoTokenizer.from_pretrained(model_name)       # ok
# model = AutoModel.from_pretrained(model_name)               # ok
# model = AlbertModel.from_pretrained(model_name)             # ok
# print('end')
# # ckiplab_bert-base-chinese
# from transformers import AutoTokenizer, AutoModel, BertTokenizer, BertModel
# model_name = '/data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/ckiplab_bert-base-chinese'
# tokenizer = BertTokenizer.from_pretrained(model_name)       # ok
# tokenizer = AutoTokenizer.from_pretrained(model_name)       # ok
# model = AutoModel.from_pretrained(model_name)               # ok
# model = BertModel.from_pretrained(model_name)             # ok
# print('end')



# # # luhua_chinese_pretrain_mrc_macbert_large
# from transformers import AutoTokenizer, AutoModel, BertTokenizer, BertModel
# model_name = '/data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/luhua_chinese_pretrain_mrc_macbert_large'
# tokenizer = BertTokenizer.from_pretrained(model_name)       # ok
# tokenizer = AutoTokenizer.from_pretrained(model_name)       # ok
# model = AutoModel.from_pretrained(model_name)               # ok
# model = BertModel.from_pretrained(model_name)             # ok
# print('end')
# # # luhua_chinese_pretrain_mrc_roberta_wwm_ext_large
# from transformers import AutoTokenizer, AutoModel, BertTokenizer, BertModel
# model_name = '/data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/luhua_chinese_pretrain_mrc_roberta_wwm_ext_large'
# tokenizer = BertTokenizer.from_pretrained(model_name)       # ok
# tokenizer = AutoTokenizer.from_pretrained(model_name)       # ok
# model = AutoModel.from_pretrained(model_name)               # ok
# model = BertModel.from_pretrained(model_name)             # ok
# print('end')



# # # uer_chinese_roberta_L-4_H-512
# from transformers import AutoTokenizer, AutoModel, BertTokenizer, BertModel
# model_name = '/data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/uer_chinese_roberta_L-4_H-512'
# tokenizer = BertTokenizer.from_pretrained(model_name)       # ok
# tokenizer = AutoTokenizer.from_pretrained(model_name)       # ok
# model = AutoModel.from_pretrained(model_name)               # ok
# model = BertModel.from_pretrained(model_name)             # ok
# print('end')
# from transformers import AutoTokenizer, AutoModel, BertTokenizer, BertModel
# model_name = '/data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/uer_roberta-base-finetuned-jd-full-chinese'
# tokenizer = BertTokenizer.from_pretrained(model_name)       # ok
# tokenizer = AutoTokenizer.from_pretrained(model_name)       # ok
# model = AutoModel.from_pretrained(model_name)               # ok
# model = BertModel.from_pretrained(model_name)             # ok
# print('end')


# # peterchou_nezha-chinese-base
from transformers import AutoTokenizer, AutoModel, BertTokenizer, BertModel
model_name = '/data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/peterchou_nezha-chinese-base'
tokenizer = BertTokenizer.from_pretrained(model_name)       # ok
tokenizer = AutoTokenizer.from_pretrained(model_name)       # ok
model = AutoModel.from_pretrained(model_name)               # ok
model = BertModel.from_pretrained(model_name)             # ok
print('end')
