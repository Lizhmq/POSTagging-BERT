import torch
from transformers import BertTokenizer
from model import POSModel, build_model


class Args:
    pass
args = Args()
args.pretrain_dir = "hfl/chinese-roberta-wwm-ext"
args.device = torch.device("cuda", 1)

model = build_model(args, 3)
tokenizer = BertTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
print(tokenizer.tokenize("测 试 测 试"))
print(tokenizer.tokenize("测试测试"))
