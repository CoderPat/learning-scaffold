for lp in "\"en-de\"" "\"en-zh\"" "\"et-en\"" "\"ne-en\"" "\"ro-en\"" "\"ru-en\"" "null"
do
    CUDA_VISIBLE_DEVICES=5 python3 meta_expl/train.py \
    --task mlqe \
    --task-params "{\"eval_lp\": ${lp}}" \
    --setup static_teacher \
    --arch xlm-r \
    --tokenizer xlm-r \
    --explainer attention_explainer \
    --explainer-params "{\"normalizer_fn\": \"softmax\", \"normalize_head_coeffs\": \"sparsemax\", \"aggregator_idx\": \"mean\", \"aggregator_dim\": \"row\", \"layer_idx\": null, \"head_idx\": null}" \
    --teacher-explainer attention_explainer \
    --teacher-explainer-params "{\"normalizer_fn\": \"softmax\", \"normalize_head_coeffs\": \"sparsemax\", \"aggregator_idx\": \"mean\", \"aggregator_dim\": \"row\", \"init_fn\": \"uniform\", \"layer_idx\": null, \"head_idx\": null}" \
    --initialize-embeddings \
    --optimizer sgd \
    --patience 5 \
    --learning-rate 5e-3 \
    --num-examples 4100 \
    --kld-coeff 5 \
    --meta-interval 1 \
    --meta-lr 1e-3 \
    --meta-warmup 0 \
    --num-resets 0 \
    --meta-explicit \
    --batch-size 16 \
    --teacher-dir notebooks/data/mlqe-xlmr-models/teacher_dir \
    --teacher-explainer-dir notebooks/data/mlqe-xlmr-models/teacher_expl_dir \
    --seed 9 \
    --wandb meta-expl
done




