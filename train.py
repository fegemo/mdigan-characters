import logging
import sys
from math import ceil

import tensorflow as tf

from util import io_utils
from util.dataset_utils import load_multi_domain_ds
from configuration import OptionParser
import setup
from model import MDIGANModel

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.DEBUG,
    datefmt="%Y-%m-%d %H:%M:%S")
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)


config, parser = OptionParser().parse(sys.argv[1:], True)
logging.info(f"Running with options: {parser.get_description(', ', ':')}")
if config.verbose:
    logging.debug(f"Tensorflow version: {tf.__version__}")

if tf.test.gpu_device_name():
    logging.info("Using this GPU: {}".format(tf.test.gpu_device_name()))
else:
    logging.warning("Not using a GPU - it will take long!!")

# check if datasets need unzipping
if config.verbose:
    logging.info("Datasets used: ", config.datasets_used)
setup.ensure_datasets(config.verbose)

# setting the seed
if config.verbose:
    logging.debug("SEED set to: ", config.seed)
tf.random.set_seed(config.seed)

# loading the dataset according to the required model
train_ds, test_ds = load_multi_domain_ds(config)

# instantiates the model
model = MDIGANModel(config)

io_utils.save_model_description(model)
if config.verbose:
    model.discriminator.summary()
    model.generator.summary()
parser.save_configuration()


# configuration for training
steps = config.steps
epochs = steps / ceil(len(train_ds) / config.batch)
evaluate_steps = config.evaluate_steps

logging.info(
    f"Starting training for {epochs:.2f} epochs in {steps} steps, updating visualization every "
    f"{evaluate_steps} steps...")
model.fit(train_ds, test_ds, steps, evaluate_steps)

logging.info(f"Saving the generator...")
io_utils.save_generator(model)

logging.info(f"Starting to generate the images from the test dataset with generator...")
io_utils.generate_images_from_dataset(model, test_ds, num_images=100)

logging.info("Finished executing.")

# Example run:
# python train.py --rm2k --steps 40000 --no-tran --lambda-l1 100 --lambda-domain 10 --lambda-ssim 10 --lr 0.0001
