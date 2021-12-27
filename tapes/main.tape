import "submitters.tape"

task TrainTeacher 
    > teacher_dir
    :: .submitter=@ .mem=16000 .gres="gpu:v100:1" .cpus=2 .time=0
    :: repo=@
    :: task=@
    :: teacher_arch=@
{
    python $repo/meta_expl/train.py \
      --task $task \
      --arch $teacher_arch \
      --tokenizer $teacher_arch \
      --model-dir $teacher_dir
}

task TrainStudent 
    < teacher_dir=@TrainTeacher
    > student_dir
    > teacher_expl_dir
    > student_expl_dir
    > results
    :: .submitter=@ .mem=16000 .gres="gpu:1" .cpus=2 .time=0
    :: repo=@
    :: task=@
    :: teacher_arch=@
    :: student_arch=@
    :: teacher_explainer=@
    :: student_explainer=@
    :: normalizer_fn=@
    :: num_epochs=@
    :: lr=@
    :: batch_size=@
    :: patience=@
    :: num_examples=@
    :: kld_coeff=@
    :: metalearn=@
    :: meta_interval=@
    :: meta_lr=@
    :: meta_explicit=@
    :: seed=@
{
    explainer_args="{\"normalizer_fn\": \"$normalizer_fn\"}"
    # TODO: add back number of epochs
    python $repo/meta_expl/train.py \
      --task $task \
      --setup $([ "$metalearn" = true ] && echo "learnable_teacher" || echo "static_teacher") \
      --arch $student_arch \
      --tokenizer $teacher_arch \
      --explainer $student_explainer \
      --explainer-params "$explainer_args" \
      --teacher-dir $teacher_dir \
      --teacher-explainer $teacher_explainer \
      --teacher-explainer-params "$explainer_args" \
      --patience $patience \
      --learning-rate $lr \
      $([ ! -z "$num_examples"  ] && echo "--num-examples $num_examples" || echo "") \
      --kld-coeff $kld_coeff \
      --meta-interval $meta_interval \
      --meta-lr $meta_lr \
      $([ "$meta_explicit" = true ] && echo "--meta-explicit" || echo "") \
      --batch-size $batch_size \
      --model-dir $student_dir \
      --teacher-explainer-dir $teacher_expl_dir \
      --explainer-dir $student_expl_dir \
      --seed $seed | tee results
}

task AverageResults
  < results=@TrainStudent[Seed:*]
  > avg_accuracy
  > avg_simulability
  :: .submitter=shell
{
  i=0
  total_accuracy=0
  total_simulability=0
  for file in $results 
  do
    accuracy=`cat $file | tail -n 1 | sed -nr 's/.*Accuracy: ([0-9\.]+).*/\1/p'`
    simulability=`cat $file | tail -n 1 | sed -nr 's/.*Simulability: ([0-9\.]+).*/\1/p'`
    echo $total_accuracy
    total_accuracy=`echo "$total_accuracy + $accuracy" | bc`
    total_simulability=`echo "$total_simulability + $simulability" | bc`
    i=$((i + 1))
  done
  echo $total_accuracy
  echo "scale=4; $total_accuracy / $i" | bc -l > $avg_accuracy
  echo "scale=4; $total_simulability / $i" | bc -l > $avg_simulability
}

summary EvalSummary {
  of AverageResults > Accuracy Simulability {
    cp $avg_accuracy $Accuracy
    cp $avg_simulability $Simulability
  }
}
