import json
from src.evals.long_context_jobs import TriviaMCQA
from torch.utils.data import DataLoader
from transformers.data.data_collator import DataCollatorWithPadding
from src.evals.finetuning_jobs import build_dataloader
from transformers import AutoTokenizer

fname = "ds5.json"
split = "train"

ds_raw = json.load(open(fname))[split]
ds = TriviaMCQA("ds5.json", split)

collator = DataCollatorWithPadding(ds.tokenizer, padding="longest", max_length=8_192)
dl = build_dataloader(ds, collator, **dict(batch_size=2))


# def col(xs):
#     return collate_padmask(xs, ds.tokenizer.pad_token_id)


# dl = DataLoader(ds, batch_size=2, collate_fn=col)
