from mmengine.config import read_base


with read_base():
    from .xft_model_utils import prepare_hf_models, add_xFT_models
    from ...models.chatglm.hf_chatglm3_6b import models, _meta_template

tokenizer_path = "/data/models/chatglm3-6b-hf"
model_path = tokenizer_path
model_path_xft = model_path + "-xft"

# fix running issue and change the dtype to bf16 for HF models.
prepare_hf_models(models, model_path, tokenizer_path)


# could be compare with the following snippet from configs/models/hf_llama:
models = add_xFT_models(
    models,
    model_path_xft,
    tokenizer_path,
    meta_template=_meta_template,
    generation_kwargs={
        "do_sample": False,
        "max_length": 4096,
    },
    dtype=[
        # "fp16",
        "bf16",
        # "int8",
        # "w8a8",
        # "int4",
        # "nf4",
        # "bf16_fp16",
        # "bf16_int8",
        # "bf16_w8a8",
        # "bf16_int4",
        # "bf16_nf4",
        # "w8a8_int8",
        # "w8a8_int4",
        # "w8a8_nf4",
    ],
    kv_cache_type=[
        # "fp32",
        "fp16",
        # "int8"
    ],
)
