import logging
import os
import shutil
import time

import tensorflow as tf
from tqdm import tqdm
from matplotlib import pyplot as plt


def ensure_folder_structure(*folders):
    is_absolute_path = os.path.isabs(folders[0])
    provided_paths = []
    for path_part in folders:
        provided_paths.extend(path_part.split(os.sep))
    folder_path = os.getcwd() if not is_absolute_path else "/"

    for folder_name in provided_paths:
        folder_path = os.path.join(folder_path, folder_name)
        if not os.path.isdir(folder_path):
            os.mkdir(folder_path)


def delete_folder(path):
    shutil.rmtree(path, ignore_errors=True)


def seconds_to_human_readable(timestamp):
    days = timestamp // 86400  # (60 * 60 * 24)
    hours = timestamp // 3600 % 24  # (60 * 60) % 24
    minutes = timestamp // 60 % 60
    seconds = timestamp % 60

    time_string = ""
    if days > 0:
        time_string += f"{days:.0f} day{'s' if days > 1 else ''}, "
    if hours > 0 or days > 0:
        time_string += f"{hours:02.0f}h:"
    time_string += f"{minutes:02.0f}m:{seconds:02.0f}s"

    return time_string


def show_eta(training_start_time, step_start_time, current_step, total_steps,
             update_steps):
    now = time.time()
    elapsed = now - training_start_time
    steps_so_far = tf.cast(current_step, tf.float32)
    elapsed_per_step = elapsed / (steps_so_far + 1.)
    remaining_steps = total_steps - steps_so_far
    eta = elapsed_per_step * remaining_steps

    logging.info(f"Time since start: {seconds_to_human_readable(elapsed)}")
    logging.info(f"Estimated time to finish: {seconds_to_human_readable(eta.numpy())}")
    logging.info(f"Last {update_steps} steps took: {now - step_start_time:.2f}s\n")


def preview_generated_images_during_training(model, examples, save_name, step):
    number_of_domains = model.config.number_of_domains
    title = [f"Input {i}" for i in range(number_of_domains)] + ["Target", "Generated"]
    num_images = len(examples)
    num_columns = len(title)

    if step is not None:
        if step == 1:
            step = 0
        title[-1] += f" ({step / 1000}k)"
        title[-2] += f" ({step / 1000}k)"

    figure = plt.figure(figsize=(4 * num_columns, 4 * num_images))

    for i, example in enumerate(examples):
        domain_images, target_domain = example

        real_image = domain_images[target_domain]
        domain_images = tf.constant(domain_images)

        # this zeroes out the target image:
        target_domain_mask = tf.one_hot(target_domain, number_of_domains, on_value=0., off_value=1.)
        domain_images *= target_domain_mask[..., tf.newaxis, tf.newaxis, tf.newaxis]

        fake_image = model.generator([tf.expand_dims(domain_images, 0), tf.expand_dims(target_domain, 0)],
                                     training=True)

        images = [*domain_images, real_image, fake_image[0]]
        for j in range(num_columns):
            idx = i * num_columns + j + 1
            plt.subplot(num_images, num_columns, idx)
            if i == 0:
                if j < number_of_domains:
                    plt.title("Input" if j != target_domain else "Target", fontdict={"fontsize": 24})
                else:
                    plt.title(title[j], fontdict={"fontsize": 24})
            elif j == target_domain:
                plt.title("Target", fontdict={"fontsize": 24})

            plt.imshow(images[j] * 0.5 + 0.5)
            plt.axis("off")

    figure.tight_layout()

    if save_name is not None:
        plt.savefig(save_name, transparent=True)

    return figure


def generate_images_from_dataset(model, dataset, num_images=None):
    dataset = dataset.unbatch()
    if num_images is None:
        num_images = dataset.cardinality()

    dataset = list(dataset.take(num_images).as_numpy_iterator())

    base_image_path = model.config.get_output_folder("test-images")

    delete_folder(base_image_path)
    ensure_folder_structure(base_image_path)

    number_of_domains = model.config.number_of_domains
    # for each image i in the dataset...
    for i, domain_images in enumerate(tqdm(dataset, total=len(dataset))):
        # for each number m of missing domains [1 to d[
        for m in range(1, number_of_domains):
            image_path = os.sep.join([base_image_path, f"{i}_missing_{m}.png"])
            fig = plt.figure(figsize=(4 * number_of_domains, 4 * number_of_domains))
            plt.suptitle(f"Missing {m} image(s)", fontdict={"size": 20})
            for target_index in range(number_of_domains):
                input_dropout_mask = tf.one_hot(target_index, number_of_domains, on_value=0., off_value=1.)

                # define which domains will be available as sources
                shuffled_domain_indices = tf.random.shuffle(tf.range(number_of_domains)).numpy().tolist()
                selected_to_drop = {target_index}
                while len(selected_to_drop) < m:
                    index_to_drop = shuffled_domain_indices.pop(0)
                    input_dropout_mask *= tf.one_hot(index_to_drop, number_of_domains, on_value=0., off_value=1.)
                    selected_to_drop.add(index_to_drop)

                # generate the image using the available sources
                dropped_input_image = domain_images * input_dropout_mask[..., tf.newaxis, tf.newaxis, tf.newaxis]
                dropped_input_image = tf.expand_dims(dropped_input_image, 0)
                target_domain = tf.expand_dims(target_index, 0)
                fake_image = model.generator([dropped_input_image, target_domain], training=True)

                for source_index in range(number_of_domains):
                    idx = (target_index * number_of_domains) + source_index + 1
                    plt.subplot(number_of_domains, number_of_domains, idx)
                    if target_index == source_index:
                        plt.title("Generated", fontdict={"fontsize": 20})
                        image = tf.squeeze(fake_image)
                    else:
                        if source_index in selected_to_drop:
                            plt.title("Dropped", fontdict={"fontsize": 20})
                        image = domain_images[source_index]

                    plt.imshow(image * 0.5 + 0.5)
                    plt.axis("off")

            plt.savefig(image_path, transparent=True)
            plt.close(fig)

    print(f"Generated {(i + 1) * number_of_domains * (number_of_domains - 1)} images in the test-images folder.")


def save_model_description(model, folder_path=None):
    if folder_path is None:
        folder_path = model.config.get_output_folder()
    ensure_folder_structure(folder_path)
    with open(os.sep.join([folder_path, "model-description.txt"]), "w") as fh:
        for network in [model.generator, model.discriminator]:
            network.summary(print_fn=lambda x: fh.write(x + "\n"))
            fh.write("\n" * 3)


def save_generator(model):
    py_model_path = model.config.get_output_folder(["saved-model"])
    delete_folder(py_model_path)
    ensure_folder_structure(py_model_path)

    model.generator.save(py_model_path)
    save_model_description(model, py_model_path)
