
VISIBLE_CUDA_DEVICES=2 python3 scripts/calc_plausiblity_mlqe.py \
    --teacher-dir notebooks/data/mlqe-xlmr-explainer/teacher_dir \
    --explainer-dir notebooks/data/mlqe-xlmr-explainer/teacher_expl_dir \
    --lp ro-en \
    --tokenizer xlm-roberta-base \
    --max-len 256 \
    --batch-size 16
