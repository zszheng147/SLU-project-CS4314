# Define the model names
model_names=("hfl/chinese-lert-base" "hfl/chinese-pert-base")

# Iterate through the model names
for model_name in ${model_names[@]}; do
    # Iterate through the learning rates
    for lr in `seq 1e-5 2e-5 5.1e-5`; do
        # Construct the command
        cmd="python scripts/slu_bert.py --device 1 --lr $lr --model_name $model_name"
        # Execute the command
        eval $cmd
    done
done
