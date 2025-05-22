from hydra import initialize, compose
from src.trainer.runner import Runner


def main():
    with initialize(version_base="1.3.2", config_path="./config/trainer"):
        cfg = compose(config_name="trainer.yaml")

    runner = Runner(cfg)
    runner.run()

if __name__ == "__main__":
    main()
