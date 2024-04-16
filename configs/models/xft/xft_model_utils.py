from opencompass.models import XFTCausalLM
from typing import Dict, List, Optional, Union
import os

DEFAULT_XFT_DTYPE_LIST = [
    "fp16",
    "bf16",
    "int8",
    "w8a8",
    "int4",
    "nf4",
    "bf16_fp16",
    "bf16_int8",
    "bf16_w8a8",
    "bf16_int4",
    "bf16_nf4",
    "w8a8_int8",
    "w8a8_int4",
    "w8a8_nf4",
]

DEFAULT_XFT_KVCACHE_DTYPE_LIST = ["fp32", "fp16", "int8"]

# This environment variable is used exclusively for testing with XFT, without testing with HF models.
ENV_XFT_ONLY_XFT = "XFT_ONLY_XFT"

# This environment variable is used exclusively for testing with HF models, without testing with xFT.
ENV_XFT_ONLY_HF = "XFT_ONLY_HF"

# XFT_DTYPE_LIST contains a list like:
# [
#   "fp16",
#   "bf16",
#   "int8",
#   "w8a8",
#   "int4",
#   "nf4",
#   "bf16_fp16",
#   "bf16_int8",
#   "bf16_w8a8",
#   "bf16_int4",
#   "bf16_nf4",
#   "w8a8_int8",
#   "w8a8_int4",
#   "w8a8_nf4",
# ]
# For example, it can include a single parameter like XFT_DTYPE_LIST=bf16
# or multiple parameters separated by commas like XFT_DTYPE_LIST=bf16,fp16,int8.
ENV_XFT_DTYPE_LIST = "XFT_DTYPE_LIST"

# XFT_KVCACHE_DTYPE_LIST environment variable contains a list of data types used for XFT KV cache.
# [
#   "fp32",
#   "fp16",
#   "int8",
# ]
# For example, it can include a single parameter like XFT_KVCACHE_DTYPE_LIST=fp16
# or multiple parameters separated by commas like XFT_KVCACHE_DTYPE_LIST=fp32,fp16,int8.
ENV_XFT_KVCACHE_DTYPE_LIST = "XFT_KVCACHE_DTYPE_LIST"

# XFT_MODEL_PATH environment variable contains path of the xFT model weight.
ENV_XFT_MODEL_PATH = "XFT_MODEL_PATH"

# XFT_HF_MODEL_PATH environment variable contains path of the HF model weight.
ENV_XFT_HF_MODEL_PATH = "XFT_HF_MODEL_PATH"

# XFT_TOKEN_PATH environment variable contains path of the tokenizer weight.
ENV_XFT_TOKEN_PATH = "XFT_TOKEN_PATH"

# XFT_MAX_NEW_LEN environment variable contains the max new length.
ENV_XFT_MAX_NEW_LEN = "XFT_MAX_NEW_LEN"


def prepare_hf_models(models, model_path, tokenizer_path):
    # disable GPU device and change the dtype to bf16

    ENV_XFT_HF_MODEL_PATH_STR = os.getenv(ENV_XFT_HF_MODEL_PATH)
    if ENV_XFT_HF_MODEL_PATH_STR:
        model_path = ENV_XFT_HF_MODEL_PATH_STR

    ENV_XFT_TOKEN_PATH_STR = os.getenv(ENV_XFT_TOKEN_PATH)
    if ENV_XFT_TOKEN_PATH_STR:
        tokenizer_path = ENV_XFT_TOKEN_PATH_STR

    max_out_len = 16
    ENV_XFT_MAX_NEW_LEN_STR = os.getenv(ENV_XFT_MAX_NEW_LEN)
    if ENV_XFT_MAX_NEW_LEN_STR:
        max_out_len = int(ENV_XFT_MAX_NEW_LEN_STR)

    for _model in models:
        _model.abbr = _model.abbr + "-bf16"
        _model.path = model_path
        _model.tokenizer_path = tokenizer_path
        _model.max_out_len = max_out_len
        _model.run_cfg.num_gpus = 0
        _model.model_kwargs.torch_dtype = "torch.bfloat16"


def add_xFT_models(
    hf_models,
    xft_model_path,
    tokenizer_path,
    meta_template: Optional[Dict] = None,
    dtype: list = DEFAULT_XFT_DTYPE_LIST,
    kv_cache_type: list = DEFAULT_XFT_KVCACHE_DTYPE_LIST,
    model_kwargs: dict = dict(),
    generation_kwargs: dict = dict(),
    **kwargs
):
    # the naming same like the HF models.
    abbr = (
        hf_models[0].abbr[:-5]
        if hf_models[0].abbr.endswith("-bf16")
        else hf_models[0].abbr
    )

    if os.getenv(ENV_XFT_ONLY_XFT):
        hf_models = []

    # got from env variables
    ENV_XFT_DTYPE_LIST_STR = os.getenv(ENV_XFT_DTYPE_LIST)
    if ENV_XFT_DTYPE_LIST_STR:
        dtype = dtype + [
            env_dtype.strip() for env_dtype in ENV_XFT_DTYPE_LIST_STR.split(",")
        ]

    ENV_XFT_KVCACHE_DTYPE_LIST_STR = os.getenv(ENV_XFT_KVCACHE_DTYPE_LIST)
    if ENV_XFT_KVCACHE_DTYPE_LIST_STR:
        kv_cache_type = kv_cache_type + [
            env_cache_type.strip()
            for env_cache_type in ENV_XFT_KVCACHE_DTYPE_LIST_STR.split(",")
        ]

    ENV_XFT_MODEL_PATH_STR = os.getenv(ENV_XFT_MODEL_PATH)
    if ENV_XFT_MODEL_PATH_STR:
        xft_model_path = ENV_XFT_MODEL_PATH_STR

    ENV_XFT_TOKEN_PATH_STR = os.getenv(ENV_XFT_TOKEN_PATH)
    if ENV_XFT_TOKEN_PATH_STR:
        tokenizer_path = ENV_XFT_TOKEN_PATH_STR

    max_out_len = 16
    ENV_XFT_MAX_NEW_LEN_STR = os.getenv(ENV_XFT_MAX_NEW_LEN)
    if ENV_XFT_MAX_NEW_LEN_STR:
        max_out_len = int(ENV_XFT_MAX_NEW_LEN_STR)

    xft_models = [
        dict(
            type=XFTCausalLM,
            abbr="{}-xft-({})-({})".format(abbr, _dtype, _kv_cache_dtype),
            path=xft_model_path,
            tokenizer_path=tokenizer_path,
            model_kwargs=dict(
                dtype=_dtype, kv_cache_dtype=_kv_cache_dtype, **model_kwargs
            ),
            tokenizer_kwargs=dict(
                padding_side="left",
                truncation_side="left",
                trust_remote_code=True,
                use_fast=False,
            ),
            meta_template=meta_template,
            max_out_len=max_out_len,
            max_seq_len=2048,
            batch_size=8,
            run_cfg=dict(num_gpus=0, num_procs=1),
            generation_kwargs={**generation_kwargs},
            batch_padding=False,
            **kwargs
        )
        for _dtype in dtype
        for _kv_cache_dtype in kv_cache_type
    ]
    if os.getenv(ENV_XFT_ONLY_HF):
        xft_models = []

    return hf_models + xft_models
