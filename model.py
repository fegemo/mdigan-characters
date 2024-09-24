import logging
import os
from abc import ABC, abstractmethod
import time

import tensorflow as tf

from util import io_utils, dataset_utils
from util.keras_utils import ConstantThenLinearDecay
from networks import collagan_affluent_generator, collagan_original_discriminator


class MDIGANModel(ABC):
    def __init__(self, config):
        self.config = config

        self.generator_optimizer = None
        self.discriminator_optimizer = None

        self.discriminator = collagan_original_discriminator(config.number_of_domains, config.image_size,
                                                             config.output_channels)
        self.generator = collagan_affluent_generator(config.number_of_domains, config.image_size,
                                                     config.output_channels, config.capacity)

        self.lambda_l1 = config.lambda_l1 or config.lambda_l1
        self.lambda_l1_backward = config.lambda_l1_backward or config.lambda_l1
        self.lambda_domain = config.lambda_domain
        self.lambda_ssim = config.lambda_ssim

        if config.input_dropout == "none":
            self.sampler = SimpleSampler(config)
        elif config.input_dropout == "original":
            self.sampler = InputDropoutSampler(config)
        elif config.input_dropout == "balanced":
            self.sampler = BalancedInputDropoutSampler(config)
        elif config.input_dropout == "conservative":
            self.sampler = ConservativeInputDropoutSampler(config)
        elif config.input_dropout == "curriculum":
            self.sampler = CurriculumLearningSampler(config)
        else:
            raise ValueError(f"The provided {config.input_dropout} type for input dropout has not been implemented.")

        if config.cycled_source_replacer in ["", "dropout"]:
            self.cycled_source_replacer = DroppedOutCycledSourceReplacer(config)
        else:
            self.cycled_source_replacer = ForwardOnlyCycledSourceReplacer(config)

        self.cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

    def fit(self, train_ds, test_ds, steps, evaluate_steps):
        # initialize generator and discriminator optimizers
        lr_generator = ConstantThenLinearDecay(self.config.lr, steps)
        lr_discriminator = ConstantThenLinearDecay(self.config.lr, steps)

        self.generator_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_generator, beta_1=0.5, beta_2=0.999)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_discriminator, beta_1=0.5,
                                                                beta_2=0.999)

        examples_for_visualization = self.select_examples_for_visualization(train_ds, test_ds)

        training_start_time = time.time()
        step_start_time = training_start_time

        for step, batch in train_ds.repeat().take(steps).enumerate():
            # every UPDATE_STEPS and in the beginning, visualize x images to see how training is going...
            it_is_time_to_evaluate = (step + 1) % evaluate_steps == 0 or step == 0 or step == steps - 1
            if it_is_time_to_evaluate:
                if step != 0:
                    io_utils.show_eta(training_start_time, step_start_time, step, steps, evaluate_steps)

                step_start_time = time.time()

                save_image_name = os.sep.join([
                    self.config.get_output_folder(),
                    "step_{:06d},update_{:03d}.png".format(step + 1, (step + 1) // evaluate_steps)
                ])
                logging.info(f"Previewing images generated at step {step + 1} (train + test)...")
                io_utils.preview_generated_images_during_training(self, examples_for_visualization,
                                                                  save_image_name, step + 1)

                logging.info(f"Step: {(step + 1) / 1000}k")
                if step < steps - 1:
                    print("_" * (evaluate_steps // 10))

            # actually TRAIN
            t = tf.cast(step / steps, tf.float32)
            self.train_step(batch, step, evaluate_steps, t)

            # dot feedback for every 10 training steps
            if (step + 1) % 10 == 0 and step < steps - 1:
                print(".", end="", flush=True)

        logging.info("About to exit the training loop...")

    def generator_loss(self, fake_predicted_patches, cycled_predicted_patches, fake_image, real_image,
                       cycled_images, source_images_5d,
                       fake_predicted_domain, cycle_predicted_domain, target_domain,
                       input_dropout_mask, batch_shape):
        number_of_domains = batch_shape[0]
        batch_size, image_size, channels = batch_shape[1], batch_shape[2], batch_shape[4]
        number_of_domains_float = tf.cast(number_of_domains, tf.float32)
        source_images = tf.reshape(source_images_5d, [batch_size * number_of_domains, image_size, image_size, channels])
        cycled_images_5d = tf.reshape(cycled_images, [batch_size, number_of_domains, image_size, image_size, channels])
        input_dropout_mask_1d = tf.reshape(input_dropout_mask, [batch_size * number_of_domains])

        # adversarial (lsgan) loss
        adversarial_forward__loss = tf.reduce_mean(tf.math.squared_difference(fake_predicted_patches, 1.))
        adversarial_backward_loss = tf.reduce_mean(tf.math.squared_difference(cycled_predicted_patches, 1.)) * \
                                    number_of_domains_float
        adversarial_loss = (adversarial_forward__loss + adversarial_backward_loss) / \
                           (number_of_domains_float + 1.)

        # l1 (forward, backward)
        l1_forward__loss = tf.reduce_mean(tf.abs(real_image - fake_image))
        l1_backward_loss = tf.reduce_mean(
            tf.reduce_sum(
                # mean of pixel l1s per image, but 0 for dropped out input images
                tf.reduce_mean(tf.abs(source_images_5d - cycled_images_5d), axis=[2, 3, 4]) * input_dropout_mask,
                axis=1),
            axis=0)

        # ssim loss (forward, backward)
        ssim_forward_ = tf.image.ssim(fake_image + 1., real_image + 1., 2)
        ssim_backward = tf.image.ssim(cycled_images + 1., source_images + 1., 2) * input_dropout_mask_1d
        # ssim_forward_ (shape=[b,])
        # ssim_backward (shape=[b*d,])
        ssim_forward__loss = tf.reduce_mean(-tf.math.log((1. + ssim_forward_) / 2.))
        ssim_backward_loss = tf.reduce_mean(tf.reduce_sum(-tf.math.log((1. + ssim_backward) / 2.)))
        ssim_loss = (ssim_forward__loss + ssim_backward_loss * number_of_domains_float) / (number_of_domains_float + 1.)

        # domain classification loss (forward, backward)
        forward__domain = tf.one_hot(target_domain, number_of_domains)
        backward_domain = tf.tile(tf.one_hot(tf.range(number_of_domains), number_of_domains), [batch_size, 1])
        backward_predicted_domain = cycle_predicted_domain
        # forward__domain (shape=[b, d])
        # backward_domain (shape=[b*d, d])

        classification_forward__loss = self.cce(forward__domain, fake_predicted_domain)
        classification_backward_loss = self.cce(backward_domain, backward_predicted_domain) * number_of_domains_float
        classification_loss = (classification_forward__loss + classification_backward_loss) / \
                              (number_of_domains_float + 1.)

        # observation: ssim loss uses only the backward (cycled) images... that's on the colla's code and paper
        total_loss = adversarial_loss + \
                     self.lambda_l1 * l1_forward__loss + self.lambda_l1_backward * l1_backward_loss + \
                     self.lambda_ssim * ssim_backward_loss + \
                     self.lambda_domain * classification_loss

        return {"total": total_loss, "adversarial": adversarial_loss, "l1_forward": l1_forward__loss,
                "l1_backward": l1_backward_loss, "ssim": ssim_loss, "domain": classification_loss}

    def discriminator_loss(self, source_predicted_patches, cycled_predicted_patches, source_predicted_domain,
                           real_predicted_patches, fake_predicted_patches, batch_shape):
        number_of_domains, batch_size = batch_shape[0], batch_shape[1]

        adversarial_real = tf.reduce_mean(tf.math.squared_difference(real_predicted_patches, 1.))
        adversarial_fake = tf.reduce_mean(tf.math.square(fake_predicted_patches))
        adversarial_forward_loss = adversarial_real + adversarial_fake

        adversarial_backward_loss = tf.reduce_mean(tf.math.squared_difference(source_predicted_patches, 1.)) + \
                                    tf.reduce_mean(tf.math.square(cycled_predicted_patches))
        adversarial_backward_loss *= tf.cast(number_of_domains, tf.float32)

        adversarial_loss = (adversarial_forward_loss + adversarial_backward_loss) / \
                           (tf.cast(number_of_domains, tf.float32) + 1.)

        source_label_domain = tf.tile(tf.one_hot(tf.range(number_of_domains), number_of_domains), [batch_size, 1])
        # source_label_domain (shape=[b*d, d])
        domain_loss = self.cce(source_label_domain, source_predicted_domain)

        total_loss = adversarial_loss + \
                     self.lambda_domain * domain_loss
        return {"total": total_loss, "real": adversarial_real, "fake": adversarial_fake, "domain": domain_loss}

    def get_cycled_images_input(self, domain_images, forward_target_domain, input_dropout_mask, fake_image,
                                batch_shape):
        """
        Returns a list of tensors that represent the input for the generator to create the cycle images
        :param domain_images: batch images for all domains (shape=[b, d, s, s, c])
        :param forward_target_domain: batched target domain index (shape=[b])
        :param input_dropout_mask: mask for which domain images have been dropped out (due to input dropout and being
        the target domain) (shape=[b, d], with 0s for dropped out images)
        :param fake_image: batched generated image (shape=[b, s, s, c])
        :param batch_shape: tuple representing the shape of the batch
        :return: a list of tensors that can be used as input to the generator so cycle images get created
        """
        number_of_domains, batch_size, image_size, channels = batch_shape[0], batch_shape[1], batch_shape[2], \
            batch_shape[4]
        backward_target_domain = tf.range(number_of_domains)
        backward_target_domain = tf.tile(backward_target_domain[tf.newaxis, ...], [batch_size, 1])
        # backward_target_domain (shape=[b, d]) with an index per image and domain in the batch

        backward_target_domain_mask = tf.one_hot(backward_target_domain, number_of_domains, on_value=0., off_value=1.)
        backward_target_domain_mask = backward_target_domain_mask[..., tf.newaxis, tf.newaxis, tf.newaxis]
        # backward_target_domain_mask (shape=[b, d_target, d, 1, 1, 1]) with 0s for images that must be suppressed
        # (as they are the target)

        # a. repeat the domain images once for each domain, so we can later have an input set with a
        # zeroed backward target for each domain
        repeated_domain_images = tf.tile(tf.expand_dims(domain_images, 1), [1, number_of_domains, 1, 1, 1, 1])
        # repeated_domain_images (shape=[b, d_target, d, s, s, c]

        # b. replace the original forward target image with the generated fake image
        # forward_target_domain_mask = tf.one_hot(forward_target_domain, number_of_domains,
        #                                         dtype=tf.bool, on_value=True, off_value=False)
        # forward_target_domain_mask = forward_target_domain_mask[:, tf.newaxis, :, tf.newaxis, tf.newaxis, tf.newaxis]
        # the mask becomes of shape [b, 1, d, 1, 1, 1], which can be broadcast to the repeated_domain_images' shape

        # b. replace the original forward target image with the generated fake image, plus the ones that have
        # been dropped out. it can be only the forward target (--cycled-source-replacer == "forward") or all the
        # source images that have been dropped out due to input dropout (--cycled-source-replacer == "dropout")
        # the Colla's paper does not specify what it does, but the source code¹ uses the "dropout" option
        # ¹ https://github.com/jongcye/CollaGAN_CVPR/blob/master/model/CollaGAN_fExp8.py#L99
        cycled_source_replacement_mask = self.cycled_source_replacer.replace(forward_target_domain, input_dropout_mask)
        fake_image = fake_image[:, tf.newaxis, tf.newaxis, ...]
        # fake_image becomes shape=[b, 1, 1, s, s, c], so it can be broadcast together with repeated_domain_images

        fake_replaced_target_domain_images = tf.where(cycled_source_replacement_mask, fake_image,
                                                      repeated_domain_images)
        # fake_replaced_target_domain_images (shape=[b, d, d, s, s, c])

        # c. zero out the images that are the backwards cyclical target
        zeroed_retarget_domain_images = fake_replaced_target_domain_images * backward_target_domain_mask

        # list of:
        # - input images with shape [b, d, d, s, s, c]
        # - input target domain with shape [b, d]

        zeroed_retarget_domain_images = tf.reshape(zeroed_retarget_domain_images, [
            batch_size * number_of_domains, number_of_domains, image_size, image_size, channels
        ])
        backward_target_domain = tf.reshape(backward_target_domain, [batch_size * number_of_domains])

        return [zeroed_retarget_domain_images, backward_target_domain]

    @tf.function
    def train_step(self, batch, step, update_steps, t):
        # [d, b, s, s, c] = domain, batch, size, size, channels
        batch_shape = tf.shape(batch)
        number_of_domains, batch_size, image_size, channels = batch_shape[0], batch_shape[1], \
            batch_shape[2], batch_shape[4]

        # 1. select a random target domain with a subset of the images as input
        domain_images, target_domain, input_dropout_mask = self.sampler.sample(batch, t)
        # domain_images (shape=[b, d, s, s, c])
        # target_domain (shape=[b,])
        # input_dropout_mask (shape=[b, d]), with 0s for images that should be dropped out

        # dropped_input_images contains, for each domain, an image or zeros with same shape (dropped out)
        dropped_input_image = domain_images * input_dropout_mask[..., tf.newaxis, tf.newaxis, tf.newaxis]
        # dropped_input_image (shape=[b, d, s, s, c], but with all 0s for dropped images)

        # real_image is the target for each example in the batch
        real_image = tf.gather(domain_images, target_domain, batch_dims=1)

        with tf.GradientTape(persistent=True) as tape:
            # --- FORWARD PASS: generate the missing image
            # 1. generate a batch of fake images
            # shape=[b, d, s, s, c]
            generator_input = [dropped_input_image, target_domain]
            # shape=[b, s, s, c]
            fake_image = self.generator(generator_input, training=True)

            # --- BACKWARD PASS: generate the cycled images using the one generated in the forward pass
            # 2. generate a batch of cycled images (back to their source domain)
            cycled_generator_input = self.get_cycled_images_input(domain_images, target_domain, input_dropout_mask,
                                                                  fake_image, batch_shape)
            # cycled_generator_input (list of [shape=[b*d, d, s, s, c], shape=[b*d]])
            cycled_images = self.generator(cycled_generator_input, training=True)
            # cycled_images (shape=[b*d, s, s, c])

            # --- DISCRIMINATION STEP
            # 3. discriminate the real (target) and fake images, then the cycled ones and the source (to train the disc)
            real_predicted_patches, real_predicted_domain = self.discriminator(real_image, training=True)
            fake_predicted_patches, fake_predicted_domain = self.discriminator(fake_image, training=True)
            # xxxx_predicted_patches (shape=[b, 1, 1, 1])
            # xxxx_predicted_domain  (shape=[b, d] -> logits)

            cycled_predicted_patches, cycled_predicted_domain = self.discriminator(cycled_images, training=True)
            # cycled_predicted_patches (shape=[b*d, 1, 1, 1])
            # cycled_predicted_domain  (shape=[b*d, d] -> logits)

            source_predicted_patches, source_predicted_domain = self.discriminator(
                tf.reshape(domain_images, [-1, image_size, image_size, channels]),
                training=True)
            # source_predicted_patches (shape=[b*d, 1, 1, 1])
            # source_predicted_domain  (shape=[b*d, d] -> logits)

            # --- LOSS COMPUTATION STEP
            # 4. calculate loss terms for the generator
            g_loss = self.generator_loss(fake_predicted_patches, cycled_predicted_patches, fake_image, real_image,
                                         cycled_images, domain_images, fake_predicted_domain,
                                         cycled_predicted_domain,
                                         target_domain, input_dropout_mask, batch_shape)

            # 5. calculate loss terms for the discriminator
            d_loss = self.discriminator_loss(source_predicted_patches, cycled_predicted_patches,
                                             source_predicted_domain, real_predicted_patches, fake_predicted_patches,
                                             batch_shape)

        # --- OPTIMIZATION STEP (gradient calculation and weight update)
        generator_gradients = tape.gradient(g_loss["total"], self.generator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(generator_gradients, self.generator.trainable_variables))

        discriminator_gradients = tape.gradient(d_loss["total"], self.discriminator.trainable_variables)
        self.discriminator_optimizer.apply_gradients(
            zip(discriminator_gradients, self.discriminator.trainable_variables))

    def select_examples_for_visualization(self, train_ds, test_ds):
        number_of_domains = self.config.number_of_domains
        number_of_examples = 3

        ensure_inside_range = lambda x: x % number_of_domains
        train_examples = []
        test_examples = []

        train_ds_iter = train_ds.unbatch().take(number_of_examples).as_numpy_iterator()
        test_ds_iter = test_ds.shuffle(self.config.test_size).unbatch().take(number_of_examples).as_numpy_iterator()
        for c in range(number_of_examples):
            target_index = ensure_inside_range(c + 1)
            train_batch = next(train_ds_iter)
            train_example = (train_batch, target_index)

            test_batch = next(test_ds_iter)
            test_example = (test_batch, target_index)

            train_examples.append(train_example)
            test_examples.append(test_example)

        return train_examples + test_examples


class ExampleSampler(ABC):
    def __init__(self, config):
        self.config = config

    @abstractmethod
    def sample(self, batch, t):
        pass

    def random_target_index(self, batch_size):
        return tf.random.uniform(shape=[batch_size], maxval=self.config.number_of_domains, dtype=tf.int32)


class InputDropoutSampler(ExampleSampler):
    def __init__(self, config):
        super().__init__(config)
        # a list shape=(d, to_drop, ?, d) that is, per possible target pose index (first dimension),
        #     for each possible number of dropped inputs (second dimension): all permutations of a boolean array that
        #     (a) nullifies the target index and (b) nullifies a number of additional inputs equal to 0, 1 or 2
        #     (determined by inputs_to_drop).
        dropout_null_list = dataset_utils.create_input_dropout_index_list([1, 2, 3], self.config.number_of_domains)
        self.null_list = tf.ragged.constant(dropout_null_list, ragged_rank=2, dtype="bool")

    def select_number_of_inputs_to_drop(self, batch_size, dropout_null_list_for_target, t):
        return tf.random.uniform(shape=[batch_size],
                                 maxval=tf.shape(dropout_null_list_for_target[0])[0],
                                 dtype="int32")

    def sample(self, batch, t):
        batch_shape = tf.shape(batch)
        number_of_domains, batch_size = batch_shape[0], batch_shape[1]

        # reorders the batch from [d, b, s, s, c] to [B, D, s, s, c]
        batch = tf.transpose(batch, [1, 0, 2, 3, 4])

        # finds a random target side for each example in the batch
        # (shape=[b,]), eg, [0, 1, 3, 3]
        target_domain_index = self.random_target_index(batch_size)

        # applies input dropout as described in the CollaGAN paper and implemented in the code
        #  this is adapted from getBatch_RGB_varInp in CollaGAN
        #  a. randomly choose an input dropout mask such as [True, False, False, True]
        #     it is done by indexing the null_list dimension by dimension until we have
        #     a tensor with a boolean dropout mask per example in the batch (shape=[b, d])
        dropout_null_list_for_target = tf.gather(
            tf.tile(self.null_list[tf.newaxis, ...], [batch_size, 1, 1, 1, 1]),
            target_domain_index, batch_dims=1)
        # dropout_null_list_for_target (shape=[b, to_drop, ?, d])
        random_number_of_inputs_to_drop = self.select_number_of_inputs_to_drop(batch_size, dropout_null_list_for_target,
                                                                               t)
        dropout_null_list_for_target_and_number_of_inputs = tf.gather(dropout_null_list_for_target,
                                                                      random_number_of_inputs_to_drop,
                                                                      batch_dims=1)
        # dropout_null_list_for_target_and_number_of_inputs (shape=[b, ?, d])
        random_permutation_index = tf.random.uniform(shape=[batch_size],
                                                     maxval=tf.shape(dropout_null_list_for_target_and_number_of_inputs)[
                                                         0], dtype="int32")
        input_dropout_mask = tf.gather(dropout_null_list_for_target_and_number_of_inputs,
                                       random_permutation_index,
                                       batch_dims=1)
        # input_dropout_mask (shape=[b, d])
        input_dropout_mask = tf.where(input_dropout_mask, 0., 1.)

        return batch, target_domain_index, input_dropout_mask


class BalancedInputDropoutSampler(InputDropoutSampler):
    def select_number_of_inputs_to_drop(self, batch_size, dropout_null_list_for_target, t):
        # 43% of the time, drop 3 inputs
        # 43% of the time, drop 2 inputs
        # 14% of the time, drop 1 inputs
        u = tf.random.uniform(shape=[batch_size])
        return tf.where(u < 0.43, 3, tf.where(u < 0.86, 2, 1)) - 1


class ConservativeInputDropoutSampler(InputDropoutSampler):
    def select_number_of_inputs_to_drop(self, batch_size, dropout_null_list_for_target, t):
        # 10% of the time, drop 3 inputs
        # 30% of the time, drop 2 inputs
        # 60% of the time, drop 1 inputs
        u = tf.random.uniform(shape=[batch_size])
        return tf.where(u < 0.1, 3, tf.where(u < 0.4, 2, 1)) - 1


class CurriculumLearningSampler(InputDropoutSampler):
    def __init__(self, config):
        super().__init__(config)
        self.balanced_sampler = BalancedInputDropoutSampler(config)

    def select_number_of_inputs_to_drop(self, batch_size, dropout_null_list_for_target, t):
        # start with easy (missing 1) samples, then move to harder ones
        # until 17% of the training, drop 1 inputs
        # until 33% of the training, drop 2 inputs
        # until 50% of the training, drop 3 inputs
        # remainder 50% of the training, drop randomly
        n = self.balanced_sampler.select_number_of_inputs_to_drop(batch_size, dropout_null_list_for_target, t) + 1
        return tf.where(t < 0.166667, 1, tf.where(t < 0.33333, 2, tf.where(t < 0.5, 3, n))) - 1


class SimpleSampler(ExampleSampler):
    def sample(self, batch, t):
        batch_shape = tf.shape(batch)
        number_of_domains, batch_size = batch_shape[0], batch_shape[1]

        # reorders the batch from [d, b, s, s, c] to [B, D, s, s, c]
        batch = tf.transpose(batch, [1, 0, 2, 3, 4])

        # finds a random target side for each example in the batch
        # (shape=[b,]), eg, [0, 1, 3, 3]
        target_domain_index = self.random_target_index(batch_size)

        # creates a mask (e {0, 1}) representing which inputs will be provided to the generator for each
        # example in the batch
        # (shape=[b, d]), eg, [[0, 1, 1, 1], ..., [1, 0, 1, 1]]
        input_dropout_mask = tf.one_hot(target_domain_index, number_of_domains)
        input_dropout_mask = tf.where(input_dropout_mask == 1., 0., 1.)

        return batch, target_domain_index, input_dropout_mask


class CycledSourceReplacer(ABC):
    def __init__(self, config):
        self.config = config

    @abstractmethod
    def replace(self, forward_target_domain, input_dropout_mask):
        pass


class ForwardOnlyCycledSourceReplacer(CycledSourceReplacer):
    def __init__(self, config):
        super().__init__(config)

    def replace(self, forward_target_domain, input_dropout_mask):
        number_of_domains = self.config.number_of_domains
        forward_target_domain_mask = tf.one_hot(forward_target_domain, number_of_domains,
                                                dtype=tf.bool, on_value=True, off_value=False)
        forward_target_domain_mask = forward_target_domain_mask[:, tf.newaxis, :, tf.newaxis, tf.newaxis, tf.newaxis]
        # the mask becomes of shape [b, 1, d, 1, 1, 1], which can be broadcast to the repeated_domain_images' shape
        return forward_target_domain_mask


class DroppedOutCycledSourceReplacer(CycledSourceReplacer):
    def __init__(self, config):
        super().__init__(config)

    def replace(self, forward_target_domain, input_dropout_mask):
        inverted_input_dropout_mask = tf.logical_not(tf.cast(input_dropout_mask, tf.bool))
        inverted_input_dropout_mask = inverted_input_dropout_mask[:, tf.newaxis, :, tf.newaxis, tf.newaxis, tf.newaxis]
        # the mask becomes of shape [b, 1, d, 1, 1, 1], which can be broadcast to the repeated_domain_images' shape

        return inverted_input_dropout_mask
