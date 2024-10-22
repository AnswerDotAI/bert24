# script to collect stats on the size of data on the contract-nli dataset
from __future__ import print_function
import statistics
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

tokenizer_name = "bclavie/olmo_bert_template"
dataset_name = "kiddothe2b/contract-nli"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)


def asciihist(it, bins=10, minmax=None, str_tag='',
              scale_output=30, generate_only=False, print_function=print):
    """Create an ASCII histogram from an interable of numbers.
    Author: Boris Gorelik boris@gorelik.net. based on  http://econpy.googlecode.com/svn/trunk/pytrix/pytrix.py
    License: MIT
    """
    ret = []
    itarray = np.asanyarray(it)
    if minmax == 'auto':
        minmax = np.percentile(it, [5, 95])
        if minmax[0] == minmax[1]:
            # for very ugly distributions
            minmax = None
    if minmax is not None:
        # discard values that are outside minmax range
        mn = minmax[0]
        mx = minmax[1]
        itarray = itarray[itarray >= mn]
        itarray = itarray[itarray <= mx]
    if itarray.size:
        total = len(itarray)
        counts, cutoffs = np.histogram(itarray, bins=bins)
        cutoffs = cutoffs[1:]
        if str_tag:
            str_tag = '%s ' % str_tag
        else:
            str_tag = ''
        if scale_output is not None:
            scaled_counts = counts.astype(float) / counts.sum() * scale_output
        else:
            scaled_counts = counts

        if minmax is not None:
            ret.append('Trimmed to range (%s - %s)' % (str(minmax[0]), str(minmax[1])))
        for cutoff, original_count, scaled_count in zip(cutoffs, counts, scaled_counts):
            ret.append("{:s}{:>8.2f} |{:<7,d} | {:s}".format(
                str_tag,
                cutoff,
                original_count,
                "*" * int(scaled_count))
                       )
        ret.append(
            "{:s}{:s} |{:s} | {:s}".format(
                str_tag,
                '-' * 8,
                '-' * 7,
                '-' * 7
            )
        )
        ret.append(
            "{:s}{:>8s} |{:<7,d}".format(
                str_tag,
                'N=',
                total
            )
        )
    else:
        ret = []
    if not generate_only:
        for line in ret:
            print_function(line)
    ret = '\n'.join(ret)
    return ret

print("Examining the distribution of input lengths in:")
print(f"Dataset: {dataset_name}")
print(f"Tokenizer: {tokenizer_name}")

for sset in ["contractnli_a","contractnli_b"]:
    print(f"\nExamining only the TRAIN split of {dataset_name}, subset {sset}")
    d = load_dataset(dataset_name,sset)
    dt = d["train"]
    pop_count = len(dt)
    print(f"n: {pop_count}")
    xs = []
    for i in range(len(dt)):
        premise = dt[i]['premise']
        hypothesis = dt[i]['hypothesis']
        in_length = len(tokenizer.encode(premise) + tokenizer.encode(hypothesis))
        xs.append(in_length)
    print("Analyzing input length (token length of d.premise + d.hypothesis):")
    print(f"Mean: {statistics.mean(xs):.2f}")
    print(f"Median: {statistics.median(xs)}")
    print(f"StDev: {statistics.stdev(xs):.2f}")
    print("Histogram: Input Length, in tokens")
    asciihist(xs, minmax=None)
    print("")
    def print_examples_in_bounds(beg,end):
        n = len([x for x in xs if beg <= x < end])
        print(f"Items with length between {beg} and {end}: {n} ({float(n)/pop_count:.2%})")
    print_examples_in_bounds(0,1_000)
    print_examples_in_bounds(1_000,2_000)
    print_examples_in_bounds(2_000,8_000)
              
print("end")
