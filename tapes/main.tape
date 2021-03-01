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
{
    CUDA_VISIBLE_DEVICES=$(get_free_gpu --is-empty) python $repo/meta_expl/train.py \
      --model-type student \
      --num-examples $num_examples \
      --kld-coeff $kld_coeff \
      --teacher-dir $teacher_dir \
      --model-dir $student_dir \
        > results
}

summary EvalSummary {
  of TrainStudent > Accuracy Simulability {
    tail -n 1 $results | grep -oP 'Accuracy: \K[0-9.]+' > $Accuracy
    tail -n 1 $results | grep -oP 'Simulability: \K[0-9.]+' > $Simulability
  }
}
