import argparse
import os
import datetime
import sys
from math import ceil
from util.io_utils import ensure_folder_structure

SEED = 42

DATASET_NAMES = ["tiny-hero", "rpg-maker-2000", "rpg-maker-xp", "rpg-maker-vxace", "miscellaneous"]
DOMAINS = ["back", "left", "front", "right"]
TRAIN_PERCENTAGE = 0.85
BATCH_SIZE = 4

IMG_SIZE = 64
INPUT_CHANNELS = 4
OUTPUT_CHANNELS = 4

LAMBDA_DOMAIN = 1.
LAMBDA_L1 = 10.
LAMBDA_SSIM = 10.
LR = 0.0001


class SingletonMeta(type):
    """
    The Singleton class can be implemented in different ways in Python. Some
    possible methods include: base class, decorator, metaclass. We will use the
    metaclass because it is best suited for this purpose.
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        """
        Possible changes to the value of the `__init__` argument do not affect
        the returned instance.
        """
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


class OptionParser(metaclass=SingletonMeta):
    def __init__(self) -> None:
        self.parser = argparse.ArgumentParser()
        self.initialized = False
        self.values = {}

    def initialize(self):
        # --- TRAINING: regarding the training of the model
        self.parser.add_argument("--lr", type=float, help="(initial) learning rate", default=LR)
        self.parser.add_argument("--batch", type=int, help="the batch size", default=BATCH_SIZE)
        self.parser.add_argument("--steps", type=int, help="number of generator update steps to train",
                                 default=10000)
        self.parser.add_argument("--evaluate-steps", type=int, help="number of generator update steps "
                                                                    "to wait until an evaluation is done", default=1000)

        # --- INPUT AND OUTPUT: regarding the input and output of the model
        self.parser.add_argument("--domains", help="domain folder names (w/o number, but in order)",
                                 default=DOMAINS, nargs="+")
        self.parser.add_argument("--image-size", help="size of squared images", default=IMG_SIZE,
                                 type=int)
        self.parser.add_argument("--output-channels", help="size of squared images",
                                 default=OUTPUT_CHANNELS, type=int)
        self.parser.add_argument("--input-channels", help="size of squared images",
                                 default=INPUT_CHANNELS, type=int)
        self.parser.add_argument("--verbose", help="outputs verbosity information",
                                 default=False, action="store_true")

        # --- OPTIMIZATION: regarding the optimization of the model such as the lambda scalers
        self.parser.add_argument("--lambda-domain", type=float, help="value for λdomain",
                                 default=LAMBDA_DOMAIN)
        self.parser.add_argument("--lambda-l1", type=float, help="value for λl1 used in λl1_forward ",
                                 default=LAMBDA_L1)
        self.parser.add_argument("--lambda-l1-backward", type=float, help="value for λl1_backward "
                                                                          "(cyclic)")
        self.parser.add_argument("--lambda-ssim", type=float, help="value for λssim",
                                 default=LAMBDA_SSIM)

        # --- AUGMENTATION: regarding the augmentation of the images (for the pose imputation, hue rotation is best)
        self.parser.add_argument("--no-aug", action="store_true", help="Disables all augmentation",
                                 default=False)
        self.parser.add_argument("--no-hue", action="store_true", help="Disables hue augmentation",
                                 default=False)
        self.parser.add_argument("--no-tran", action="store_true", help="Disables translation "
                                                                        "augmentation",  default=False)

        # --- CONFIGURATION of changes atop the original CollaGAN architecture
        self.parser.add_argument("--capacity", type=int, help="capacity multiplier for the generator",
                                 default=4)
        self.parser.add_argument("--input-dropout", help="applies dropout to the input as in the "
                                                         "CollaGAN paper. Can be one from {none, "
                                                         "original, curriculum, conservative (default)}",
                                 default="conservative")
        self.parser.add_argument("--cycled-source-replacer",
                                 help="one from {dropout, forward (default)} indicating which images should be replaced"
                                      "by the forward generated one when computing the cycled images. Colla's paper"
                                      "does not specify this, but its code shows that it replaces all that have been"
                                      "dropped out", default="forward")

        # --- DATASET: regarding the datasets to use for training and evaluation
        self.parser.add_argument("--rmxp", action="store_true", default=False,
                                 help="Uses RPG Maker XP dataset")
        self.parser.add_argument("--rm2k", action="store_true", default=False,
                                 help="Uses RPG Maker 2000 dataset")
        self.parser.add_argument("--rmvx", action="store_true", default=False,
                                 help="Uses RPG Maker VX Ace dataset")
        self.parser.add_argument("--tiny", action="store_true", default=False,
                                 help="Uses the Tiny Hero dataset")
        self.parser.add_argument("--misc", action="store_true", default=False,
                                 help="Uses the miscellaneous sprites dataset")

        self.initialized = True

    def parse(self, args=None, return_parser=False):
        if args is None:
            args = sys.argv[1:]
        if not self.initialized:
            self.initialize()
        self.values = self.parser.parse_args(args)

        if self.values.lambda_l1_backward is None:
            self.values.lambda_l1_backward = self.values.lambda_l1 / 10.
        setattr(self.values, "number_of_domains", len(self.values.domains))
        setattr(self.values, "seed", SEED)
        if self.values.no_aug:
            setattr(self.values, "no_hue", True)
            setattr(self.values, "no_tran", True)
        datasets_used = list(filter(lambda opt: getattr(self.values, opt), ["tiny", "rm2k", "rmxp", "rmvx", "misc"]))
        setattr(self.values, "datasets_used", datasets_used)
        if len(datasets_used) == 0:
            raise Exception("No dataset was supplied with: --tiny, --rm2k, --rmxp, --rmvx, --misc")
        setattr(self.values, "dataset_names", DATASET_NAMES)
        setattr(self.values, "data_folders", [
            os.sep.join(["datasets", folder])
            for folder
            in self.values.dataset_names
        ])
        setattr(self.values, "domain_folders", [f"{i}-{name}" for i, name in enumerate(self.values.domains)])
        setattr(self.values, "run_string", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

        dataset_mask = list(
            map(lambda opt: 1 if getattr(self.values, opt) else 0, ["tiny", "rm2k", "rmxp", "rmvx", "misc"]))
        dataset_sizes = [912, 216, 294, 408, 12372]
        dataset_sizes = [n * m for n, m in zip(dataset_sizes, dataset_mask)]
        train_sizes = [ceil(n * TRAIN_PERCENTAGE) for n in dataset_sizes]
        train_size = sum(train_sizes)
        test_sizes = [dataset_sizes[i] - train_sizes[i]
                      for i, n in enumerate(dataset_sizes)]
        test_size = sum(test_sizes)

        setattr(self.values, "dataset_sizes", dataset_sizes)
        setattr(self.values, "dataset_mask", dataset_mask)
        setattr(self.values, "train_sizes", train_sizes)
        setattr(self.values, "train_size", train_size)
        setattr(self.values, "test_sizes", test_sizes)
        setattr(self.values, "test_size", test_size)

        self.values.epochs = ceil(self.values.steps * self.values.batch / self.values.train_size)

        setattr(self.values, "inner_channels", min(self.values.input_channels, self.values.output_channels))

        setattr(self.values, "get_output_folder", self.get_output_folder)

        if return_parser:
            return self.values, self
        else:
            return self.values

    def get_output_folder(self, sub_folder=None):
        folders = ["output", self.values.run_string]
        if sub_folder is not None:
            if not isinstance(sub_folder, list):
                sub_folder = [sub_folder]
            folders += sub_folder

        return os.sep.join(folders)

    def get_description(self, param_separator=",", key_value_separator="-"):
        sorted_args = sorted(vars(self.values).items())
        description = param_separator.join(map(lambda p: f"{p[0]}{key_value_separator}{p[1]}", sorted_args))
        return description

    def save_configuration(self, folder_path=None):
        if folder_path is None:
            folder_path = self.get_output_folder()
        ensure_folder_structure(folder_path)
        with open(os.sep.join([folder_path, "configuration.txt"]), "w") as file:
            file.write(self.get_description("\n", ": ") + "\n")
