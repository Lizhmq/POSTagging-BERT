import numpy as np
import torch



def get_sent(st):
    return [t.split("/")[0] for t in st.split()]

def get_sent_test(st):
    return [t for t in st.split()]

def get_label(st):
    return [t.split("/")[1] for t in st.split()]

def get_se(sent, block_size):
    ls = [0] + list(map(len, sent))
    st = np.cumsum(ls)
    st += 1     # [CLS]
    end = st[1:]
    st = st[:-1]
    end = list(filter(lambda x: x <= block_size, end))
    st = st[:len(end)]
    return st, end


def normalize_hiddens(hiddens, starting_offsets, ending_offsets):
    cum_attn = hiddens.cumsum(1)
    start_v = cum_attn.gather(1, starting_offsets - 1)
    end_v = cum_attn.gather(1, ending_offsets - 1)
    return end_v - start_v




if __name__ == "__main__":
    se, end = get_se(['1','22','333'], 300)
    print(se)
    print(normalize_hiddens(torch.tensor([[0,1,2,3,4,5,6,]]), torch.tensor([se]), torch.tensor([end])))
