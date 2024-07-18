from transformers import AutoTokenizer
from tokenizers import Tokenizer, processors
import os
import tempfile

import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description="Convert and extend tokenizer")
    parser.add_argument(
        "--original-tokenizer",
        type=str,
        required=True,
        help="The name or path of the original tokenizer to be extended",
    )
    parser.add_argument(
        "--target-repo-id",
        type=str,
        required=True,
        help="The repository ID where the extended tokenizer will be uploaded",
    )
    return parser.parse_args()


args = parse_arguments()


class CustomTokenizer:
    def __init__(self, pretrained_model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
        self.tokenizer.padding_side = "right"
        self.tokenizer.truncation_side = "right"

    def __getattr__(self, name):
        return getattr(self.tokenizer, name)

    def add_special_tokens(self, *args, **kwargs):
        return self.tokenizer.add_special_tokens(*args, **kwargs)

    def add_tokens(self, *args, **kwargs):
        return self.tokenizer.add_tokens(*args, **kwargs)

    def apply_template(self):
        template = processors.TemplateProcessing(
            single="[CLS] $A [SEP]",
            pair="[CLS] $A [SEP] $B [SEP]",
            special_tokens=[
                ("[CLS]", self.tokenizer.convert_tokens_to_ids("[CLS]")),
                ("[SEP]", self.tokenizer.convert_tokens_to_ids("[SEP]")),
                ("[MASK]", self.tokenizer.convert_tokens_to_ids("[MASK]")),
                ("[UNK]", self.tokenizer.convert_tokens_to_ids("[UNK]")),
                ("[PAD]", self.tokenizer.convert_tokens_to_ids("[PAD]")),
            ],
        )
        self.tokenizer.backend_tokenizer.post_processor = template

    def save_and_reload(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            self.tokenizer.save_pretrained(tmpdirname)
            self.tokenizer = AutoTokenizer.from_pretrained(tmpdirname)

    def __call__(self, *args, **kwargs):
        return self.tokenizer(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)


# Initialize and configure the tokenizer
tokenizer = CustomTokenizer(args.original_tokenizer)

og_size = len(tokenizer.vocab)
assert og_size % 32 == 0, "Original tokenizer size must be a multiple of 32"
target_size = og_size + 128

extra_toks = 128
tokenizer.add_special_tokens(
    {"unk_token": "[UNK]", "cls_token": "[CLS]", "sep_token": "[SEP]", "pad_token": "[PAD]", "mask_token": "[MASK]"},
    replace_additional_special_tokens=True,
)
tokenizer.add_special_tokens({"bos_token": "[CLS]", "eos_token": "[SEP]"})
tokenizer.add_bos_token = False
tokenizer.add_eos_token = False
unused = target_size - len(tokenizer.vocab)
tokenizer.add_tokens([f"[unused{i}]" for i in range(0, unused)])

# Apply the template
tokenizer.apply_template()

# Save and reload the tokenizer to ensure the changes are persistent
tokenizer.save_and_reload()

# Test the tokenizer
encoded = tokenizer("Hello world", "How are you?")
decoded = tokenizer.decode(encoded["input_ids"])
print(decoded)

tokenizer.push_to_hub(args.target_repo_id)
