from hydra import initialize, compose
from src.trainer.runner import Runner
import sys



def main():


    with initialize(version_base="1.3.2", config_path="./config/trainer"):
        cfg = compose(config_name="trainer.yaml")

    # Override config based on command line arguments
    if len(sys.argv) > 1:
        cfg.env_id = sys.argv[1]

    print("Environment: ", cfg.env_id)
    runner = Runner(cfg)
    runner.run()

if __name__ == "__main__":
    main()
