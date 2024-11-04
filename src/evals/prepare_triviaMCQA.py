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
import math, itertools, re, random, statistics, os, string, sys, argparse

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

def make_mcqa_dict_old(x:dict,answers:list):
    "Returns a multiple-choice Q&A dict, from the trivia_qa item x"
    evidence = wiki_context(x)
    orig_question = question(x)
    true_answer = answer(x)
    multichoice_count = 5
    if len(answers) < multichoice_count:
        raise "error"
    true_answer_idx = random.randint(0,multichoice_count-1)
    answers[true_answer_idx] = true_answer
    prompt = f"""Please carefully review the following textual evidence, which contains information relevant to answering the multiple-choice question below:

{evidence}

Based solely on the information provided in the evidence above, select the single best answer to the following question:

Question: {orig_question}

A. {answers[0]}
B. {answers[1]}
C. {answers[2]}
D. {answers[3]}
E. {answers[4]}

Respond only with the letter (A, B, C, D, or E) corresponding to the most accurate answer. Do not include any explanation or additional text in your response.

Answer:"""
    mc_answer_val = dict(enumerate(string.ascii_uppercase))[true_answer_idx]
    d = dict(
        question_id=x['question_id'],
        question=orig_question,
        context=evidence,
        qd_prompt=prompt,
        options=answers[0:multichoice_count],
        answer=mc_answer_val,
        answer_index=true_answer_idx,
            )
    return d


def make_mcqa_dict(x:dict,answers:list):
    "Returns a multiple-choice Q&A dict, from the trivia_qa item x. Mimics MC style from truthfulqa"
    evidence = wiki_context(x)
    orig_question = question(x)
    true_answer = answer(x)
    multichoice_count = 5
    if len(answers) < multichoice_count:
        raise "error"
    true_answer_idx = random.randint(0,multichoice_count-1)
    answers[true_answer_idx] = true_answer
    prompt = f"""Please carefully review the following textual evidence, which contains information relevant to answering the question below:

    ## Evidence:
    {evidence}

    ## Question
    {orig_question}"""
    
    mc_answer_val = dict(enumerate(string.ascii_uppercase))[true_answer_idx]
    d = dict(
        question_id=x['question_id'],
        question=orig_question,
        context=evidence,
        qd_prompt=prompt,
        options=answers[0:multichoice_count],
        answer=mc_answer_val,
        answer_index=true_answer_idx,
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

def update_fastdata_for_vertexai(fd:FastData):
    "Mutates a fastdata instance to use vertexai if possible."
    try:
        fd.cli:claudette.Client = vertexauth.get_claudette_client(vertex_model='claude-3-5-sonnet-v2@20241022')
        return fd
    except Exception:
        print("Unable to authenticate with VertexAI, so using default Fastdata configuration of Claudette")
        return fd


def make_mcqa_filtering(ds_split, max_mcqa_in_len=8_000):
    """
    Generates MCQA pairs from items in DS_SPLIT, filtering.
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
    idxs_to_generate = [idx for idx 
                        in range(len(filtered_ds)) 
                        if est_input_lens[idx] <= max_mcqa_in_len]
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
    update_fastdata_for_vertexai(fd)                                              
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

def main():
    parser = argparse.ArgumentParser()
    parser.description='Generates multiple-choice TriviaQA variants'
    parser.add_argument('--items', type=int, required=True, help='number of items to collect from each split')
    parser.add_argument('--output', type=Path, required=True, help='output file name')
    args = parser.parse_args()
    item_count = args.items
    out_path = args.output
    ds = load_triviaqa()
    print("loaded triviaqa")
    result = {}
    for split in ds.keys():
        dssplit = ds[split]
        xs = make_mcqa_filtering(dssplit.select(range(item_count)))
        result[split] = xs
    
    with open(str(out_path),'w') as f:
        import json
        json.dump(result,f)


if __name__ == '__main__':
    main()


