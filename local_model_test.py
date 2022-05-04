import torch
from transformers import BertTokenizer, AlbertModel
from transformers import AutoConfig

tokenizer = BertTokenizer.from_pretrained('/Users/chang/workspace_of_python/pretrained_language_model/huggingface_pretrained_models/clue_albert_chinese_tiny/')
model = AlbertModel.from_pretrained("/Users/chang/workspace_of_python/pretrained_language_model/huggingface_pretrained_models/clue_albert_chinese_tiny/")
print('ok')


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
