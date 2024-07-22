# Preload libiomp5.so by following cmd or LD_PRELOAD=libiomp5.so manually
export $(python -c 'import xfastertransformer as xft; print(xft.get_env())')

# /data/models/Qwen-14B-Chat-xft
TOKEN_PATH=/data/models/Qwen-72B-Chat
MODEL_PATH=/data/models/Qwen-72B-Chat-xft

# numactl -C 0-47 -l python -m vllm.entrypoints.openai.api_server \
#         --model ${MODEL_PATH} \
#         --tokenizer ${TOKEN_PATH} \
#         --dtype bf16 \
#         --kv-cache-dtype fp16 \
#         --served-model-name xft \
#         --port 8000 \
#         --trust-remote-code

OMP_NUM_THREADS=80 mpirun \
        -n 1 numactl --all -C 0-79 -m 0 \
          python -m vllm.entrypoints.openai.api_server \
            --model ${MODEL_PATH} \
            --tokenizer ${TOKEN_PATH} \
            --dtype fp16 \
            --kv-cache-dtype fp16 \
            --served-model-name xft \
            --port 8000 \
            --trust-remote-code \
        : -n 1 numactl --all -C 80-159 -m 1 \
          python -m vllm.entrypoints.slave \
            --dtype fp16 \
            --model ${MODEL_PATH} \
            --kv-cache-dtype fp16
