# se quiser correr: renormalize gradients by their norm to get probas

for seed in 0 9 11 30 69
do
  CUDA_VISIBLE_DEVICES=7 python3 smat/train.py \
      --task mlqe \
      --setup static_teacher \
      --arch xlm-r \
      --tokenizer xlm-r \
      --explainer attention_explainer \
      --explainer-params "{\"normalizer_fn\": \"softmax\", \"normalize_head_coeffs\": \"sparsemax\", \"aggregator_idx\": \"mean\", \"aggregator_dim\": \"row\", \"layer_idx\": null, \"head_idx\": null}" \
      --teacher-explainer gradient_norm_explainer \
      --teacher-explainer-params "{\"normalizer_fn\": \"softmax\", \"ord\": 2}" \
      --initialize-embeddings \
      --optimizer sgd \
      --patience 5 \
      --learning-rate 5e-3 \
      --num-examples 4100 \
      --kld-coeff 5 \
      --meta-interval 1 \
      --meta-lr 1e-3 \
      --warmup-steps 0 \
      --num-resets 0 \
      --batch-size 8 \
      --teacher-dir /home/mtreviso/meta-expl/saved-models/mlqe-xlmr-models/teacher_dir \
      --model-dir /home/mtreviso/meta-expl/saved-models/mlqe-xlmr-models/student_dir \
      --explainer-dir /home/mtreviso/meta-expl/saved-models/mlqe-xlmr-models/student_expl_dir \
      --seed ${seed} \
      --wandb meta-expl \
      --do-save
done


for seed in 0 9 11 30 69
do
  CUDA_VISIBLE_DEVICES=7 python3 smat/train.py \
      --task mlqe \
      --setup static_teacher \
      --arch xlm-r \
      --tokenizer xlm-r \
      --explainer attention_explainer \
      --explainer-params "{\"normalizer_fn\": \"softmax\", \"normalize_head_coeffs\": \"sparsemax\", \"aggregator_idx\": \"mean\", \"aggregator_dim\": \"row\", \"layer_idx\": null, \"head_idx\": null}" \
      --teacher-explainer gradient_norm_explainer \
      --teacher-explainer-params "{\"normalizer_fn\": \"average\", \"ord\": 2}" \
      --initialize-embeddings \
      --optimizer sgd \
      --patience 5 \
      --learning-rate 5e-3 \
      --num-examples 4100 \
      --kld-coeff 5 \
      --meta-interval 1 \
      --meta-lr 1e-3 \
      --warmup-steps 0 \
      --num-resets 0 \
      --batch-size 8 \
      --teacher-dir /home/mtreviso/meta-expl/saved-models/mlqe-xlmr-models/teacher_dir \
      --model-dir /home/mtreviso/meta-expl/saved-models/mlqe-xlmr-models/student_dir \
      --explainer-dir /home/mtreviso/meta-expl/saved-models/mlqe-xlmr-models/student_expl_dir \
      --seed ${seed} \
      --wandb meta-expl \
      --do-save
done
