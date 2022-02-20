#!/bin/bash

#echo -n "What model do you want to use? "

PS3='Which model to use: '
options=("GPT-NeoX-20B" "BERT")

model
modelname

select opt in "${options[@]}"
do
    case $opt in
        "GPT-NeoX-20B")
            echo "Using $opt model"
            model="gpt-neox"
            modelname="GPT-NeoX-20B"
            break
            ;;
        "BERT")
            echo "Using $opt model"
            model="bert"
            modelname="BERT"
            break
            ;;
        *) echo "invalid option $REPLY";;
    esac
done

while true; do
    read -p "Do you wish to install the model? [Y/n] " yn
    case $yn in
        [Yy]* ) ./.hpc/gpt-neox/setup.sh; break;;
        [Nn]* ) break;;
        * ) echo "Please answer yes or no.";;
    esac
done

sbatch ./.hpc/${model}/job.slurm
