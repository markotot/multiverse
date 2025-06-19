from hydra import initialize, compose
from src.trainer.runner import Runner
import sys

# TODO:
#   Take a trained agent and run it in a world model and qualitatively look at the results.



def main():


    with initialize(version_base="1.3.2", config_path="./config/trainer"):
        cfg = compose(config_name="trainer.yaml")

    # Override config based on command line arguments
    if len(sys.argv) > 1:
        cfg.env_id = sys.argv[1]

    print("Environment: ", cfg.env_id)
    runner = Runner(cfg)
    runner.evaluate_agent(envs=runner.eval_envs)

if __name__ == "__main__":
    main()
