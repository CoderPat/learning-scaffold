task TrainTeacher 
    > teacher_dir
    :: repo=@
{
    CUDA_VISIBLE_DEVICES=$(get_free_gpu --is-empty) python $repo/meta_expl/train.py \
      --model-dir $teacher_dir
}

task TrainStudent 
    < teacher_dir=@TrainTeacher
    > student_dir
    > results
    :: repo=@
    :: num_examples=@
    :: kld_coeff=@
    :: seed=@
{
    CUDA_VISIBLE_DEVICES=$(get_free_gpu --is-empty) python $repo/meta_expl/train.py \
      --model-type student \
      --num-examples $num_examples \
      --kld-coeff $kld_coeff \
      --teacher-dir $teacher_dir \
      --model-dir $student_dir \
      --seed $seed \
        > results
}

task AverageResults
  < results=@TrainStudent[Seed:*]
  > avg_accuracy
  > avg_simulability
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
