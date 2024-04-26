import argparse
from trainer import Trainer
from utils.config import get_cfg


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    return cfg


def main(args):
    cfg = setup(args)
    print(cfg)

    # trainer = Trainer(cfg)
    # trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", required=True, default=None, metavar="FILE", help="path to config file")
    parser.add_argument("--opts", default=None, nargs=argparse.REMAINDER, help="options to override config file")
    args = parser.parse_args()
    main(args)