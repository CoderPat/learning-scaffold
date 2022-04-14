import "submitters.tape"

task TrainTeacher 
    > teacher_dir
    :: .submitter=@ .mem=16000 .gres="gpu:1" .cpus=2 .time=0 .exclude=@
    :: repo=@
    :: task=@
    :: teacher_arch=@
    :: wandb_project=@
{
    train_params=""
    if [ "$task" == "imdb" ]; then
        train_params+="--clip-grads 1.0 "
        train_params+="--warmup-steps 500 "
    fi
    if [ "$task" == "mlqe" ]; then
        train_params+="--learning-rate 1e-5 "
        train_params+="--clip-grads 1.0 "
        train_params+="--warmup-steps 4000 "
    fi
    python $repo/smat/train.py \
      --wandb $wandb_project \
      --task $task \
      --arch $teacher_arch \
      --tokenizer $teacher_arch \
      $train_params \
      --model-dir $teacher_dir \
}

task TrainStudent 
    < teacher_dir=@TrainTeacher
    > student_dir
    > teacher_expl_dir
    > student_expl_dir
    > results
    > test_outputs
    :: .submitter=@ .mem=32000 .gres="gpu:1" .cpus=2 .time=0 .exclude=@
    :: repo=@
    :: task=@
    :: teacher_arch=@
    :: student_arch=@
    :: teacher_explainer=@
    :: student_explainer=@
    :: init_embed=@
    :: teacher_normalizer_fn=@
    :: student_normalizer_fn=@
    :: teacher_init_fn=@
    :: normalize_head_coeffs=@
    :: teacher_aggregator_idx=@
    :: student_aggregator_idx=@
    :: teacher_layer_idx=@
    :: student_layer_idx=@
    :: aggregator_normalizer_fn=@
    :: lr=@
    :: batch_size=@
    :: patience=@
    :: student_optimizer=@
    :: num_examples=@
    :: kld_coeff=@
    :: metalearn=@
    :: meta_interval=@
    :: meta_lr=@
    :: implicit_differentiation=@
    :: wandb_project=@
    :: seed=@
{
    # define teacher_explainer params
    teacher_explainer_args="\"normalizer_fn\": \"$teacher_normalizer_fn\""
    if [ "$teacher_explainer" == "attention_query_explainer" ]; then
      teacher_explainer_args+=", \"normalize_head_coeffs\": $normalize_head_coeffs"
      teacher_explainer_args+=", \"init_fn\": \"$teacher_init_fn\""
      teacher_explainer_args+=", \"layer_idx\": $teacher_layer_idx"
    elif [ "$teacher_explainer" == "attention_explainer" ]; then
      teacher_explainer_args+=", \"normalize_head_coeffs\": $normalize_head_coeffs"
      teacher_explainer_args+=", \"aggregator_idx\": $teacher_aggregator_idx"
      teacher_explainer_args+=", \"init_fn\": \"$teacher_init_fn\""
      teacher_explainer_args+=", \"layer_idx\": $teacher_layer_idx"
    fi
    teacher_explainer_args="{$teacher_explainer_args}"

    # define student_explainer params
    student_explainer_args="\"normalizer_fn\": \"$student_normalizer_fn\""
    if [ "$student_explainer" == "attention_query_explainer" ]; then
      student_explainer_args+=", \"normalize_head_coeffs\": $normalize_head_coeffs}"
      student_explainer_args+=", \"layer_idx\": $student_layer_idx"
    elif [ "$student_explainer" == "attention_explainer" ]; then
      student_explainer_args+=", \"normalize_head_coeffs\": $normalize_head_coeffs"
      student_explainer_args+=", \"aggregator_idx\": $student_aggregator_idx"
      student_explainer_args+=", \"layer_idx\": $student_layer_idx"
    fi
    student_explainer_args="{$student_explainer_args}"

    # HACK 
    export HF_HOME=/scratch/pfernand/hf_home
    export HF_DATASETS_CACHE=/scratch/pfernand/hf_datasets_cache
    
    # TODO: add back number of epochs
    python $repo/smat/train.py \
      --wandb $wandb_project \
      --task $task \
      --setup $([ "$metalearn" = true ] && echo "learnable_teacher" || echo "static_teacher") \
      --arch $student_arch \
      --tokenizer $teacher_arch \
      --explainer $student_explainer \
      --explainer-params "$student_explainer_args" \
      --teacher-dir $teacher_dir \
      --teacher-explainer $teacher_explainer \
      --teacher-explainer-params "$teacher_explainer_args" \
      $([ "$init_embed" = true ] && echo "--initialize-embeddings" || echo "") \
      --optimizer $student_optimizer \
      --patience $patience \
      --learning-rate $lr \
      $([ ! -z "$num_examples"  ] && echo "--num-examples $num_examples" || echo "") \
      --kld-coeff $kld_coeff \
      --meta-interval $meta_interval \
      --meta-lr $meta_lr \
      --meta-warmup $meta_warmup \
      --num-resets $num_resets \
      $([ "$implicit_differentiation" = true ] && echo "--implicit-differentiation" || echo "") \
      --batch-size $batch_size \
      --model-dir $student_dir \
      --teacher-explainer-dir $teacher_expl_dir \
      --explainer-dir $student_expl_dir \
      --save-test-outputs $test_outputs \
      --log-teacher-params teacher_expl_params.txt \
      --seed $seed | tee results
}

task AverageResults
  < results=@TrainStudent[Seed:*]
  < test_outputs=@TrainStudent[Seed:*]
  > mean_metric
  > median_metric
  > minmax_metric
  > iqd_metric
  > std_metric
  > bootstrap_mean_metric
  > confint_metric
  :: .submitter=slurm .mem=8000 .cpus=2 .time=0
  :: repo=@
  :: metric=@
{
  python $repo/smat/bootstrap_eval.py \
    $test_outputs \
    --metric $metric > result

  cat result | grep -oP "^Mean: \K[0-9]+\.[0-9]+" > $mean_metric
  cat result | grep -oP "^Median: \K[0-9]+\.[0-9]+" > $median_metric
  cat result | grep -oP "^Min/Max: \K.*" > $minmax_metric
  cat result | grep -oP "^STD: \K.*" > $std_metric
  cat result | grep -oP "^q25/q75: \K.*" > $iqd_metric
  cat result | grep -oP "^Bootstrapped Mean: \K.*" > $bootstrap_mean_metric
  cat result | grep -oP "^Confidence Interval: .*\K\[.*\]" > $confint_metric
  
}

summary EvalSummary {
  of AverageResults > PaperStr MeanSim MedianSim MinMaxSim IQDSim STDSim BootstrapMeanSim ConfIntervalSim {
    cp $mean_metric $MeanSim
    cat $median_metric | sed -r "s/0\.([0-9]{2})([0-9]{2})[0-9]*/\1.\2/" > $MedianSim
    cp $minmax_metric $MinMaxSim
    cat $iqd_metric | sed -r "s/0\.([0-9]{2})([0-9]{2})[0-9]*\/0\.([0-9]{2})([0-9]{2})[0-9]*/\[\1.\2:\3.\4\]/" > $IQDSim
    paste -d '+' $MedianSim $IQDSim > $PaperStr
    cp $std_metric $STDSim
    cp $bootstrap_mean_metric $BootstrapMeanSim
    cp $confint_metric $ConfIntervalSim
  }
}
