import json
from src.evals.long_context_jobs import TriviaMCQA
from src.evals.long_context_jobs import collate_padmask
from torch.utils.data import DataLoader

from transformers import AutoTokenizer

fname = "ds5.json"
split = "train"


ds_raw = json.load(open(fname))[split]
ds = TriviaMCQA("ds5.json",split)

def col(xs):
    return collate_padmask(xs,ds.tokenizer.pad_token_id)

dl = DataLoader(ds,batch_size=2,collate_fn=col)

