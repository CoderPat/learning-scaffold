
for seed in 0 9 11 30 69
do
  for num_ex in 4500 2250 9000
  do
    CUDA_VISIBLE_DEVICES=7 python3 smat/train.py \
      --task cifar100 \
      --setup static_teacher \
      --arch vit-base \
      --tokenizer vit-base \
      --explainer attention_explainer \
      --explainer-params "{\"normalizer_fn\": \"softmax\", \"normalize_head_coeffs\": \"sparsemax\", \"aggregator_idx\": \"mean\", \"aggregator_dim\": \"row\", \"layer_idx\": null, \"head_idx\": null}" \
      --teacher-explainer gradient_norm_explainer \
      --teacher-explainer-params "{\"normalizer_fn\": \"softmax\", \"ord\": 2}" \
      --initialize-embeddings \
      --optimizer sgd \
      --patience 5 \
      --learning-rate 5e-3 \
      --num-examples ${num_ex} \
      --kld-coeff 0.2 \
      --meta-interval 1 \
      --meta-lr 5e-3 \
      --warmup-steps 0 \
      --num-resets 0 \
      --batch-size 16 \
      --teacher-dir /home/mtreviso/meta-expl/saved-models/cifar100-vit-models/teacher_dir \
      --seed ${seed} \
      --wandb meta-expl
  done
done
