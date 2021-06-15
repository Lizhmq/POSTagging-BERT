import os
import pickle
import torch
from torch.utils.data import Dataset


class ClassifierDataset(Dataset):
    def __init__(self, tokenizer, args, logger, file_name, map_dict=None, block_size=300):
        if args.local_rank == -1:
            local_rank = 0
            world_size = 1
        else:
            local_rank = args.local_rank
            world_size = torch.distributed.get_world_size()

        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        cached_file = os.path.join(args.output_dir, file_name[:-4] +"_blocksize_%d" %
                                   (block_size)+"_wordsize_%d" % (world_size)+"_rank_%d" % (local_rank))
        if os.path.exists(cached_file):
            with open(cached_file, 'rb') as handle:
                datas = pickle.load(handle)
                self.inputs, self.labels = datas["x"], datas["y"]
                self.starts, self.ends = datas["start"], datas["end"]
                self.map_dict = datas["dict"]
        else:
            self.inputs, self.labels = [], []
            self.starts, self.ends = [], []
            datafile = os.path.join(args.data_dir, file_name)
            datas = pickle.load(open(datafile, "rb"))
            inputs, labels = datas["x"], datas["y"]
            starts, ends = datas["start"], datas["end"]
            length = len(inputs)

            if map_dict == None:
                label_set = set()
                for label in labels:
                    for lab in label:
                        label_set.add(lab)
                self.map_dict = dict()
                for lab in label_set:
                    if lab not in self.map_dict:
                        self.map_dict[lab] = len(self.map_dict)
            else:
                self.map_dict = map_dict

            for idx, (data, label, start, end) in enumerate(zip(inputs, labels, starts, ends)):
                if idx % world_size == local_rank:
                    start, end = list(start), list(end)
                    code = " ".join(data)
                    code_tokens = tokenizer.tokenize(code)[:block_size-2]
                    code_tokens = [tokenizer.cls_token] + \
                        code_tokens + [tokenizer.sep_token]
                    code_ids = tokenizer.convert_tokens_to_ids(code_tokens)

                    padding_length = block_size - len(code_ids)
                    code_ids += [tokenizer.pad_token_id] * padding_length
                    
                    label = label[:len(start)]
                    y_padding = block_size - len(label)
                    # if idx == 0:
                    #     print(label)
                    label = [self.map_dict[tt] for tt in label]
                    label += [-100] * y_padding
                    
                    st_padding = block_size - len(start)
                    start += [start[-1] - 1 for _ in range(st_padding)]
                    end += [start[-1] - 1 for _ in range(st_padding)]
                    
                    
                    self.inputs.append(code_ids)
                    self.labels.append(label)
                    self.starts.append(start)
                    self.ends.append(end)

                # if idx % (length // 10) == 0:
                #     percent = idx / (length//10) * 10
                #     logger.warning("Rank %d, load %d" % (local_rank, percent))

            with open(cached_file, 'wb') as handle:
                pickle.dump({"x": self.inputs, "y": self.labels, "start": self.starts, "end": self.ends, "dict": self.map_dict},
                            handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, item):
        return torch.tensor(self.inputs[item]), torch.tensor(self.labels[item]), \
                 torch.tensor(self.starts[item]), torch.tensor(self.ends[item])
