import argparse
import logging

import numpy as np
from numpy.core.arrayprint import set_printoptions
import torch
from transformers import (AdamW, BertTokenizer, get_linear_schedule_with_warmup)
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler)
from tqdm import tqdm
from dataset import ClassifierDataset
from model import POSModel, build_model
from utils import get_sent, get_label, get_se


logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


def load_file(file, block_size=512):
    with open(file, "r", encoding="utf-16") as f:
        data = f.readlines()
    data = data[:int(0.1 * len(data))]
    xs, sts, ends = [], [], []
    for line in data:
        sent = get_sent(line)
        st, end = get_se(sent, block_size)
        if len(sent) == 0:
            continue
        xs.append(sent)
        # ys.append(label)
        sts.append(st)
        ends.append(end)
    return {"x": xs, "start": sts, "end": ends}


def generate_prediction(model, tokenizer, data, label_dic, args):
    xs = data["x"]
    starts, ends = data["start"], data["end"]
    block_size = args.block_size
    final_preds = []
    inv_map = dict()
    for key, value in label_dic.items():
        inv_map[value] = key
    for idx, (x, st, en) in tqdm(enumerate(zip(xs, starts, ends))):
        x = " ".join(x)
        x = tokenizer.tokenize(x)[:block_size-2]
        x = [tokenizer.cls_token] + x + [tokenizer.sep_token]
        x_ids = tokenizer.convert_tokens_to_ids(x)

        padding_length = block_size - len(x_ids)
        x_ids += [tokenizer.pad_token_id] * padding_length

        # y = y[:len(st)]
        # y = [label_dic[tt] for tt in y]
        y_padding = block_size - len(st)
        # y += [-100] * y_padding

        st = list(st)   # numpy array bad for list concat
        en = list(en)
        st += [st[-1] - 1 for _ in range(y_padding)]
        en += [en[-1] - 1 for _ in range(y_padding)]

        this_len = len(st)
        x_ids = torch.tensor([x_ids]).to(args.device)
        # y = torch.tensor([y]).to(args.device)
        st = torch.tensor([st]).to(args.device)
        en = torch.tensor([en]).to(args.device)

        x_mask = x_ids.ne(tokenizer.pad_token_id).to(x_ids)
        lens = torch.LongTensor([this_len]).to(x_ids)

        # only one sample
        pred = model.predict(x_ids, x_mask, st, en, lens)[0, :].cpu().numpy()
        pred = [inv_map[v] for v in pred[:lens]]
        final_preds.append(pred)
    return final_preds


def output_file(preds, data, out_file):
    xs = data["x"]
    lss_cnt = 0
    with open(out_file, "w") as f:
        for idx, (pred, x) in enumerate(zip(preds, xs)):
            curlen = len(x)
            if len(pred) < curlen:
                pred += ["Nc"] * (curlen - len(pred))
                lss_cnt += 1
            f.writelines("\t".join([x[i] + "/" + pred[i] for i in range(curlen)] + "\n"))
    print(f"lss_cnt: {lss_cnt}", flush=True)
    return



def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--output_dir", default=None, type=str, required=True)    
    parser.add_argument("--test_name", default=None, type=str, required=True,       # load for label_map
                        help="The test data name.")    
    parser.add_argument("--input_file", default=None, type=str, required=True,
                        help="The input data path.")
    parser.add_argument("--output_file", default=None, type=str, required=True,
                        help="The output data path.")
    parser.add_argument("--load_dir", default=None, type=str, required=True,
                        help="The dir of pretrained model.")
    parser.add_argument("--block_size", default=300, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")
    # parser.add_argument("--eval_batch_size", default=32, type=int,
    #                     help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--seed', type=int, default=2233,
                        help="random seed for initialization")

    args = parser.parse_args()
    args.local_rank = -1
    args.device = torch.device("cpu")
    # args.device = torch.device("cuda", 0)

    model_path = args.load_dir
    in_file = args.input_file
    out_file = args.output_file
    args.pretrain_dir = model_path
    tokenizer = BertTokenizer.from_pretrained(args.pretrain_dir)
    test_name = args.test_name
    test_dataset = ClassifierDataset(tokenizer, args, logger, file_name=test_name, block_size=args.block_size)
    print(test_dataset.map_dict)
    bio_size = len(test_dataset.map_dict)
    model = build_model(args, bio_size, args.pretrain_dir)
    dic = test_dataset.map_dict
    
    data = load_file(in_file)
    preds = generate_prediction(model, tokenizer, data, dic, args)
    output_file(preds, data, out_file)

if __name__ == "__main__":
    main()