import "submitters.tape"

task TrainTeacher 
    > teacher_dir
    :: .submitter=@ .mem=16000 .gpus=1 .cpus=2
    :: repo=@
{
    python $repo/meta_expl/train.py \
      --model-dir $teacher_dir
}

task TrainStudent 
    < teacher_dir=@TrainTeacher
    > student_dir
    > teacher_expl_dir
    > student_expl_dir
    > results
    :: .submitter=@ .mem=16000 .gpus=1 .cpus=2
    :: repo=@
    :: teacher_explainer=@
    :: student_explainer=@
    :: num_examples=@
    :: kld_coeff=@
    :: metalearn=@
    :: meta_interval=@
    :: meta_lr=@
    :: meta_init=@
    :: pivot=@
    :: seed=@
{
    python $repo/meta_expl/train.py \
      --model-type student \
      --teacher-explainer $teacher_explainer \
      --student-explainer $student_explainer \
      --num-examples $num_examples \
      --kld-coeff $kld_coeff \
      $([ "$metalearn" = true ] && echo "--metalearn-interval $meta_interval --meta-lr $meta_lr --meta-init $meta_init" || echo "") \
      $([ "$pivot" = true ] && echo "--pivot" || echo "") \
      --teacher-dir $teacher_dir \
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
