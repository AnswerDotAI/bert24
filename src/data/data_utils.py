

MDS_COLS_TOKENIZED = {
        'input_ids': 'ndarray',
        'attention_mask': 'ndarray',
        'id': 'str'
}

MDS_COLS_TEXT = {
        'text': 'str',
        'id': 'str'
}


SOURCE_MAP = {
    "books": "orionweller/books_mds_incremental",
    "wiki": "orionweller/wikipedia_mds_incremental",
    "falcon": "orionweller/refinedweb_mds_incremental",
    "c4": "orionweller/c4_mds_incremental",
    "cc_en_head": "orionweller/cc_en_head_mds_incremental",
    "cc_en_tail": "orionweller/cc_en_tail_mds_incremental",
    "cc_en_middle": "orionweller/cc_en_middle_mds_incremental",
    "megawika": "orionweller/megawika_mds_incremental",
    "cc_news": "orionweller/cc_news_mds_incremental",
    "pes2o": "orionweller/pes2o_mds_incremental",
    "tulu_flan": "orionweller/tulu_flan_mds_incremental",
    "starcoder": "orionweller/starcoder_mds_incremental",
    "stackexchange": "orionweller/stackexchange_mds_incremental",
    "arxiv": "orionweller/arxiv_mds_incremental",
    "open_web_math_train": "orionweller/open-web-math_mds_incremental",
    "reddit": "orionweller/reddit_mds_incremental",
    "algebraic_stack_train": "orionweller/algebraic-stack_mds_incremental",
    "caselaw-access-project": "orionweller/caselaw-access-project",
    "fineweb-edu-10B": "orionweller/fineweb-edu-10B",
    # "fineweb-edu-350B": "orionweller/fineweb-edu-350B",
}


ALL_REPOS = list(SOURCE_MAP.values())