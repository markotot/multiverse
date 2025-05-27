# Description: Run the evaluation of the Iris model on the Atari environment

START_SEED=$1
END_SEED=$2
ENV_NAME=$3
WANDB_API_KEY=$4

wandb login $WANDB_API_KEY

echo "Running multiverse parameters:"
echo "  START_SEED: $START_SEED"
echo "  END_SEED: $END_SEED"
echo "  ENV_NAME: $ENV_NAME"

python3 -m run_experiment $ENV_NAME