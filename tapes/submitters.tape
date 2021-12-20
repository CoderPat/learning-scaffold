submitter cuda_async :: COMMANDS TASK_VARIABLES CONFIGURATION {
    action run > exit_code {
        # define script (random suffix for parallel acess)
        suffix=$((1 + $RANDOM % 1024))
        wrapper="script_${suffix}.sh"

        echo "#!/bin/bash" > $wrapper
        echo "set -e # stop on errors" >> $wrapper
        echo "set -o pipefail # stop on pipeline errors" >> $wrapper
        echo "set -u # stop on undeclared variables" >> $wrapper

        # print actual jobs
        echo "$COMMANDS" >> $wrapper
        CUDA_VISIBLE_DEVICES=$(get_free_gpu --is-empty) bash $wrapper
        rm -f $wrapper
    }
}
