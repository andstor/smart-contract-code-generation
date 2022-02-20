#!/bin/bash

#echo -n "What model do you want to use? "

PS3='Which model to use: '
options=("GPT-NeoX-20B" "BERT")

select opt in "${options[@]}"
do
    case $opt in
        "GPT-NeoX-20B")
            echo "Using $opt model"
            model="gpt-neox"
            break
            ;;
        "BERT")
            echo "Using $opt model"
            model="bert"
            break
            ;;
        *) echo "invalid option $REPLY";;
    esac
done

venv="./models/${model}/.venv"
if [ -e ${venv} ]; then
   echo "Installing dependencies"
   ./.hpc/gpt-neox/setup.sh
fi

sbatch ./.hpc/${model}/job.slurm
