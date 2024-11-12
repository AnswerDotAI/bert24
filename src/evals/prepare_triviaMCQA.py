#!/usr/bin/env -S uv run python
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "claudette",
#     "datasets",
#     "python-fastdata",
#     "fastcore",
#     "vertexauth",
#     "transformers",
# ]
# ///
### Make the line below the first line, if you don't want to depend on uv
#!/usr/bin/env -S python3
from pathlib import Path
from datasets import load_dataset
import math, itertools, re, random, statistics, os, string, sys, argparse, json
from datasets import Dataset, DatasetDict

from tqdm.contrib.concurrent import process_map  # or thread_map
from functools import partial
from transformers import AutoTokenizer
from fastdata.core import FastData
from fastcore.utils import BasicRepr, store_attr
import claudette
import vertexauth

def load_triviaqa():
    return load_dataset('mandarjoshi/trivia_qa','rc.wikipedia')

## accessors

def wiki_context(x):
    "Generates MCQA `context` from a TriviaQA items"
    ss = x['entity_pages']['wiki_context']
    sep_token = "\n\n==================================================\n\n"
    retval = sep_token.join(ss)
    return retval
def question(x): return x['question']
def answer(x): return x['answer']['value']
def question_toklen(x): return len( tokenizer.encode( question(x) ))
def wiki_toklen(x): return len( tokenizer.encode( wiki_context(x) ))
def answer_toklen(x): return len( tokenizer.encode (answer(x)))

os.environ['TOKENIZERS_PARALLELISM']='true'
tokenizer = AutoTokenizer.from_pretrained("bclavie/olmo_bert_template")

##
## 
##

def make_mcqa_dict(x:dict,answers:list):
    "Returns a MCQA item, from a trivia_qa item, given bad answers. "
    evidence = wiki_context(x)
    orig_question = question(x)
    true_answer = answer(x)
    multichoice_count = 5
    if len(answers) < multichoice_count:
        raise "error"
    true_answer_idx = random.randint(0,multichoice_count-1)
    answers[true_answer_idx] = true_answer
    mc_answer_val = dict(enumerate(string.ascii_uppercase))[true_answer_idx]
    d = dict(
        question_id=x['question_id'],
        question=orig_question,
        context=evidence,
        options=answers[0:multichoice_count],
        answer_index=true_answer_idx,
        answer=mc_answer_val,
            )
    return d
##
## data synthesis
##


class FakeTriviaQA(BasicRepr):
    "Fake, but plausible trivia answers to trivia questions"
    def __init__(
        self,
        answers: list[str], # a minimum of 10 plausible answers
    ): store_attr()
FakeTriviaQA(["a1","a2","a3", "a4", "a5", "a6", "a7", "a8", "a9", "a10"])

# prompt for generating the incorrect Q&A answers
prompt_template = """
<evidence>
{evidence}
</evidence>

<trivia_question>
{question}
</trivia_question>

<correct_answer>
{answer}
</correct_answer>

Given the above evidence, trivia_question, and the question's correct_answer, generate a minimum of 10 plausible trivia answers.

<important>
NOTE: Make sure that the answers are incorrect, but plausible.
</important>
"""

def compute_prompt_overhead_len(): 
    return len(tokenizer.encode(re.sub(r'{.*?}','',prompt_template)))

def input_length_of_mcqa_item(ds_item:dict,
                      prompt_overhead_len = compute_prompt_overhead_len()):
    "estimates MCQA input length, given an trivia_qa item"
    multichoice_count = 5
    q_len = question_toklen( ds_item )
    evidence_len = wiki_toklen( ds_item )
    multichoice_len = answer_toklen( ds_item ) * multichoice_count # estimate bad answer len ~= true answer
    return q_len + evidence_len + multichoice_len + prompt_overhead_len

def update_fastdata_for_vertexai(fd:FastData,max_mcqa_in_len):
    "Mutates a fastdata instance to use vertexai if possible."
    toks_per_call = max_mcqa_in_len
    # GCloud/VertexAI quotas
    # "Online prediction tokens per minute per base model per minute per region per base_model"
    toks_per_minute_limit = 1_630_000
    # "Online prediction requests per base model per minute per region per base_model"
    requests_per_minute_limit = 270
    # results
    toks_derived_calls_per_minute = int(toks_per_minute_limit / toks_per_call)
    final_calls_per_minute = int(0.85 * min(toks_derived_calls_per_minute, requests_per_minute_limit))
    try:
        vertex_claudette_cli = vertexauth.get_claudette_client(vertex_model='claude-3-5-sonnet-v2@20241022')
        fd.cli = vertex_claudette_cli
        fd.set_rate_limit(calls=final_calls_per_minute,period=60)
        print(f"Authenticated to VertexAI. Using VertexAI to access Anthropic models. Rate limit at {final_calls_per_minute} calls/min")
        return fd
    except Exception:
        print("Unable to authenticate with VertexAI, so using default Fastdata configuration of Claudette")
        return fd


def make_mcqa_filtering(ds_split, max_mcqa_in_len=8_000):
    """
    Generates MCQA items from TriviaQA items in DS_SPLIT, filtering.
    max_item_count : defines max count of items to assess, which may be much greater than max of total output.
    """
    multichoice_count=5
    idxs_to_estimate = range(len(ds_split))
    print(f"Estimating MCQA input lengths of {len(idxs_to_estimate)} items")
    # TODO: don't use tqdm to drive the paralellization.
    filtered_ds = ds_split.select(idxs_to_estimate)
    os.environ['TOKENIZERS_PARALLELISM']='true'
    est_input_lens = process_map(
        input_length_of_mcqa_item,      # or thread_map
        filtered_ds,            # your collection of items
        max_workers=14,         # number of CPU cores to use
        chunksize=100,        # optional: tune this for better performance
        desc="Processing to estimate input lengths"      # description for progress bar
    )
    print("Using input lengths to filter to subset of items to use for generating MCQA answers")
    # filter to questions where the MCQA input would not be too large
    idxs_to_generate = [sampled_idxs for sampled_idxs 
                        in range(len(filtered_ds)) 
                        if est_input_lens[sampled_idxs] <= max_mcqa_in_len]
    print(f"Filtered down to {len(idxs_to_generate)} items with length < {max_mcqa_in_len}")
    print("Preparing to generate fake answers")
    filtered_ds = filtered_ds.select(idxs_to_generate)
    # generate the fake answers
    toks_per_minute_limit = 400_000
    reqs_per_minute_limit = 4_000
    toks_per_call = max_mcqa_in_len
    calls_per_minute = int(0.85 * toks_per_minute_limit / toks_per_call)
    fd = FastData(model='claude-3-5-sonnet-20241022',
                  calls=min(calls_per_minute,reqs_per_minute_limit),
                  period=60)  
    update_fastdata_for_vertexai(fd,max_mcqa_in_len)
    fakes = fd.generate(
        prompt_template=prompt_template,
        inputs=[
            {"evidence": wiki_context(x),
             "question": question(x),
             "answer":answer(x)}
            for x in filtered_ds
            ],
            schema=FakeTriviaQA,
            max_workers=10)
    # filter to MCQA items where there are enough incorrect fake answers
    assert len(filtered_ds) == len(fakes), "Generated fake count did not match filtered ds count"
    valid_idxs = []
    for i in range(len(filtered_ds)):
        if not fakes[i]:
            continue
        # verify every FakeTriviaQA is type valid.
        if not isinstance(fakes[i].answers,list):
            continue
        item = filtered_ds[i]
        correct_answer = answer(item)
        valid_bad_answers = [a for a in fakes[i].answers if a != correct_answer]
        if len(valid_bad_answers) < multichoice_count:
            print(f"did not generate enough incorrect answers for item {i} in {ds_split}, so not generating an MCQA pair")
            continue
        else:
            valid_idxs.append(i)
    # assert: valid_idxs indexes into filtered_ds and fakes, and has only valid bad answers with good length
    retval = [ make_mcqa_dict(filtered_ds[i], fakes[i].answers) for i in valid_idxs]
    print(f"Generated {len(retval)} valid MCQA pairs")
    print("done")
    return retval

usage = """
In the TriviaQA dataset, every item has these keys:
- ['question', 'question_id', 'question_source', 'entity_pages', 'search_results', 'answer']

The TriviaMCQA dataset is formed by taking every question, producing a set of multiple choices, where one
is the correct answer and the others are incorrect. It uses these keys, similar to the MMLU_Pro dataset:

- question_id       :: the original feld from TriviaQA
- question          :: the original field from TriviaQA
- context:str       :: a str of evidence, which can be used to answer the question, derived from TriviaQA entity_pages
- options:list[str] :: list of possible choices for the answer
- answer_index:int  :: the zero-based index of the correct answer
- answer:str        :: the correct answer, from among the choices in `options`

"""

def main():
    parser = argparse.ArgumentParser()
    parser.description='Generates TriviaMCQA, the multiple-choice variant of TriviaQA'
    parser.add_argument('--items', type=int, required=True, help='number of items to collect from each split')
    parser.add_argument('--seed', default=42,
                        type=int, required=False,help='seed used to randomize sampling from the TriviaQA dataset')
    parser.add_argument('--output', type=Path, help='output file name')
    parser.add_argument('--push', action='store_true', help='push to HF')
    args = parser.parse_args()
    item_count = args.items
    if not args.output and not args.push:
        print("Must specify --output or --push")
        sys.exit(1)

    ds = load_triviaqa()
    print("loaded triviaqa")
    result = {}
    random.seed(args.seed)
    for split in ds.keys():
        print(f"Working on split={split}")
        dssplit = ds[split]
        randomize = True
        if randomize:
            sampled_idxs = sorted(random.sample(list(range(len(dssplit))),
                                                min(item_count,len(dssplit))))
        else:
            sampled_idxs = range(min(item_count,len(dssplit)))
        xs = make_mcqa_filtering(dssplit.select(sampled_idxs))
        result[split] = xs

    if (out_path := args.output) is not None:
        with open(str(out_path),'w') as f:
            json.dump(result,f)
    elif args.push:
        dataset = DatasetDict({split: Dataset.from_list(result[split]) for split in ["train", "validation", "test"]})
        dataset.push_to_hub("answerdotai/trivia_mcqa")
    else:
        # unreachable
        print("should be unreachable")
        sys.exit(1)

if __name__ == '__main__':
    main()


