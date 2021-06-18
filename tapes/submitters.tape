submitter slurm :: COMMANDS TASK_VARIABLES CONFIGURATION {
    action run > exit_code {
        # define script (random suffix for parallel acess)
        suffix=$((1 + $RANDOM % 1024))
        wrapper="slurm_script_${suffix}.sh"

        echo "#!/bin/bash" > $wrapper
        echo "set -e # stop on errors" >> $wrapper
        echo "set -o pipefail # stop on pipeline errors" >> $wrapper
        echo "set -u # stop on undeclared variables" >> $wrapper

        # FIXME
        echo 'export HF_HOME=/scratch/pfernand/hf' >> $wrapper

        # print actual jobs 
        echo "$COMMANDS" >> $wrapper
        srun --mem=$mem --gres=gpu:$gpus -c $cpus --exclude=compute-0-26 --job-name=$TASK --time=0 bash $wrapper
        rm -f $wrapper
    }
}