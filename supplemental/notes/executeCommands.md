## Smoke Test Init

DATA_PATH=/workspace/parameter-golf/data/datasets/fineweb10B_customdp1024_80 \
TOKENIZER_KIND=custom_jsonl \
TOKENIZER_PATH=/workspace/parameter-golf/vocab/vocab_best.jsonl \
VOCAB_SIZE=1024 \
RUN_ID=boot_custom_v1_phraseinit_1gpu \
CUSTOM_INIT_MODE=phrase_comp \
CUSTOM_INIT_BLEND=0.70 \
CUSTOM_INIT_SCALE=1.00 \
CUSTOM_INIT_MIN_PARTS=2 \
CUSTOM_INIT_MAX_PARTS=16 \
CUSTOM_INIT_ALLOW_BYTE_FALLBACK=1 \
CUSTOM_INIT_LOG_SAMPLES=20 \
WARMUP_STEPS=0 \
VAL_LOSS_EVERY=20 \
TRAIN_LOG_EVERY=5 \
MAX_WALLCLOCK_SECONDS=30 \
torchrun --standalone --nproc_per_node=1 train_gpt_custom_v2_init_locked.py 