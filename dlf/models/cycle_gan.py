import tensorflow as tf
from tensorflow.keras.layers import Add, Input, Conv2D, Activation, Dropout, Conv2DTranspose
from tensorflow.keras.layers import UpSampling2D, LeakyReLU, Concatenate
from tensorflow.keras.models import Model
from tensorflow_addons.layers.normalizations import InstanceNormalization

from dlf.core.registry import register_model
from dlf.core.model import ModelWrapper

# based on https://github.com/eriklindernoren/Keras-GAN/blob/master/cyclegan/cyclegan.py


@register_model('cycle_gan', 'CycleGAN')
class CycleGAN(ModelWrapper):
    """A CycleGAN implementation

    This class contains a CycleGAN implemenation with an U-Net architecture as generator.
    Next to the default implementation of a cycle adversarial loss, mentioned in CycleGAN paper,
    this implementations allows it to add an optional perceptual loss function.

    # Aliases
        - cycle_gan
        - CycleGAN

    # Args
        input_shape: tuple(int, int , int). Input shape of this network
        generator: str {'unet', 'resnet'}. Defines the generator type. Defaults to unet
        generator_filters: int, optional.
            Number of CNN filters used for generator. Defaults to 32.
        discriminator_filters: int, optional.
            Number of CNN filters used for discriminator. Defaults to 64.
        use_perceptual_loss: bool, optional.
            If true, in addition to the CycleGAN implementated losses the perceptual loss is used.
            Defaults to True.
        cycle_weight: float, optional. Weight for cycle consistency loss. Defaults to 10.0.
        identity_weigh: float, optional. Weight for identity loss Defaults to 1.0.
        model_weights: str, optional. Path to the pretrained model weights. Defaults to None.
        summary: bool, optional. If true a summary of the model will be printed to stdout. Defaults to False.
        optimizer: list of dict, optional. Name of optimizer used for training.
        resnet_blocks: int. Only available if generator is resnet. Defaults to 9.

    # Raises
        ValueError: If not exactly four optimizer are specified

    # YAML Configuration
    ```yaml
    model:
        cycle_gan:
            input_shape:
                - 512
                - 512
                - 3
            generator: unet
            generator_filters: 32
            discriminator_filters: 64
            use_perceptual_loss: False
            summary: True
            optimizer:
                - Adam:
                    learning_rate: 0.0002
                    beta_1: 0.5
                - Adam:
                    learning_rate: 0.0002
                    beta_1: 0.5
                - Adam:
                    learning_rate: 0.0002
                    beta_1: 0.5
                - Adam:
                    learning_rate: 0.0002
                    beta_1: 0.5

    ```

    # References
        - CycleGAN: https://arxiv.org/abs/1703.10593
        - U-Net: https://arxiv.org/abs/1505.04597
        - Perceptual Loss: https://cs.stanford.edu/people/jcjohns/papers/eccv16/JohnsonECCV16.pdf
        - https://www.tensorflow.org/tutorials/generative/cyclegan
        - https://github.com/eriklindernoren/Keras-GAN/blob/master/cyclegan/cyclegan.py
    """

    def __init__(self, input_shape, generator='unet', generator_filters=32, discriminator_filters=64,
                 use_perceptual_loss=True, cycle_weight=10.0, identity_weight=1.0, model_weights=None,
                 summary=False, optimizer=None, resnet_blocks=9, **kwargs):
        self.input_shape = input_shape
        self.use_perceptual_loss = use_perceptual_loss
        self.generator_filters = generator_filters
        self.discriminator_filters = discriminator_filters
        self.cycle_weight = cycle_weight
        self.identity_weight = identity_weight

        patch = int(self.input_shape[0] / 2**4)
        self.disc_patch = (patch, patch, 1)

        self.discriminator_X = self._build_discriminator("discriminator_X")
        self.discriminator_Y = self._build_discriminator("discriminator_Y")
        if generator == 'unet':
            self.generator_XY = self._build_unet_generator("generator_XY")
            self.generator_YX = self._build_unet_generator("generator_YX")
        else:
            self.generator_XY = self._build_resnet_generator(
                "generator_XY", n_resnet=resnet_blocks)
            self.generator_YX = self._build_resnet_generator(
                "generator_YX", n_resnet=resnet_blocks)

        img_X = Input(shape=input_shape, name="input_img_X")
        img_Y = Input(shape=input_shape, name="input_img_Y")

        fake_Y = self.generator_XY(img_X)
        fake_X = self.generator_YX(img_Y)

        reconstruct_X = self.generator_YX(fake_Y)
        reconstruct_Y = self.generator_XY(fake_X)

        image_X_id = self.generator_YX(img_X)
        image_Y_id = self.generator_XY(img_Y)

        # self.discriminator_X.trainable = False
        # self.discriminator_Y.trainable = False

        valid_X = self.discriminator_X(fake_X)
        valid_Y = self.discriminator_Y(fake_Y)

        model = Model(inputs=[img_X, img_Y], outputs=[
            valid_X, valid_Y, reconstruct_X, reconstruct_Y, image_X_id, image_Y_id], name="combined_model")

        if summary:
            model.summary()
            self.discriminator_X.summary()
            self.discriminator_Y.summary()
            self.generator_XY.summary()
            self.generator_YX.summary()

        super().__init__(model, None, optimizer, None, **kwargs)
        self.binary_cross = tf.keras.losses.BinaryCrossentropy(
            from_logits=True)
        # override super losses
        self.losses = [
            self._discriminator_loss,
            self._generator_loss,
            self._calc_cycle_loss,
            self._calc_cycle_loss,
            self._identity_loss,
            self._identity_loss,
        ]

        if len(self.optimizer) != 4:
            raise ValueError("CycleGan requires four optimizer!")

    def _build_discriminator(self, name):
        init = tf.keras.initializers.RandomNormal(stddev=0.02)

        def d_layer(layer_input, filters, f_size=4, normalization=True):
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=f_size, kernel_initializer=init,
                       strides=2, padding='same')(layer_input)
            if normalization:
                d = InstanceNormalization()(d)
            d = LeakyReLU(alpha=0.2)(d)
            return d

        img = Input(shape=self.input_shape, name="input_" + name)

        d1 = d_layer(img, self.discriminator_filters, normalization=False)
        d2 = d_layer(d1, self.discriminator_filters * 2)
        d3 = d_layer(d2, self.discriminator_filters * 4)
        d4 = d_layer(d3, self.discriminator_filters * 8)

        validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)

        return Model(img, validity, name=name)

    def _build_unet_generator(self, name):
        init = tf.keras.initializers.RandomNormal(stddev=0.02)

        def conv2d(layer_input, filters, f_size=4):
            """Layers used during downsampling"""
            d = Conv2D(filters, kernel_size=f_size, kernel_initializer=init,
                       strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            d = InstanceNormalization()(d)
            return d

        def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
            """Layers used during upsampling"""
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(filters, kernel_size=f_size, strides=1,
                       kernel_initializer=init, padding='same', activation='relu')(u)
            if dropout_rate:
                u = Dropout(dropout_rate)(u)
            u = InstanceNormalization()(u)
            u = Concatenate()([u, skip_input])
            return u

        # Image input
        d0 = Input(shape=self.input_shape, name="input_" + name)

        # Downsampling
        d1 = conv2d(d0, self.generator_filters)
        d2 = conv2d(d1, self.generator_filters * 2)
        d3 = conv2d(d2, self.generator_filters * 4)
        d4 = conv2d(d3, self.generator_filters * 8)

        # Upsampling
        u1 = deconv2d(d4, d3, self.generator_filters * 4)
        u2 = deconv2d(u1, d2, self.generator_filters * 2)
        u3 = deconv2d(u2, d1, self.generator_filters)

        u4 = UpSampling2D(size=2)(u3)
        output_img = Conv2D(self.input_shape[-1], kernel_size=4, kernel_initializer=init,
                            strides=1, padding='same', activation='tanh')(u4)
        return Model(d0, output_img, name=name)

    def _build_resnet_generator(self, name, n_resnet=9):
        init = tf.keras.initializers.RandomNormal(stddev=0.02)

        def resnet_block(n_filters, input_layer):
            # first layer convolutional layer

            g = Conv2D(n_filters, (3, 3), padding='same',
                       kernel_initializer=init)(input_layer)
            g = InstanceNormalization(axis=-1)(g)
            g = Activation('relu')(g)

            # second convolutional layer
            g = Conv2D(n_filters, (3, 3), padding='same',
                       kernel_initializer=init)(g)
            g = InstanceNormalization(axis=-1)(g)

            # concatenate merge channel-wise with input layer
            g = Add()([g, input_layer])
            return g

        in_image = Input(shape=self.input_shape)

        g = Conv2D(self.generator_filters, (7, 7), padding='same',
                   kernel_initializer=init)(in_image)
        g = InstanceNormalization(axis=-1)(g)
        g = Activation('relu')(g)

        g = Conv2D(self.generator_filters * 2, (3, 3), strides=(2, 2),
                   padding='same', kernel_initializer=init)(g)
        g = InstanceNormalization(axis=-1)(g)
        g = Activation('relu')(g)

        g = Conv2D(self.generator_filters * 4, (3, 3), strides=(2, 2),
                   padding='same', kernel_initializer=init)(g)
        g = InstanceNormalization(axis=-1)(g)
        g = Activation('relu')(g)

        for _ in range(n_resnet):
            g = resnet_block(self.generator_filters * 4, g)

        g = Conv2DTranspose(self.generator_filters * 2, (3, 3), strides=(
            2, 2), padding='same', kernel_initializer=init)(g)
        g = InstanceNormalization(axis=-1)(g)
        g = Activation('relu')(g)
        g = Conv2DTranspose(self.generator_filters, (3, 3), strides=(
            2, 2), padding='same', kernel_initializer=init)(g)
        g = InstanceNormalization(axis=-1)(g)
        g = Activation('relu')(g)
        g = Conv2D(3, (7, 7), padding='same', kernel_initializer=init)(g)
        g = InstanceNormalization(axis=-1)(g)
        out_image = Activation('tanh')(g)
        # define model
        model = Model(in_image, out_image, name=name)
        return model

    def _discriminator_loss(self, real, generated):
        real_loss = self.binary_cross(tf.ones_like(real), real)
        generated_loss = self.binary_cross(tf.zeros_like(generated), generated)
        total_disc_loss = real_loss + generated_loss
        return total_disc_loss * 0.5

    def _generator_loss(self, generated):
        return self.binary_cross(tf.ones_like(generated), generated)

    def _calc_cycle_loss(self, real_image, cycled_image):
        loss = tf.reduce_mean(tf.abs(real_image - cycled_image))
        return self.cycle_weight * loss

    def _identity_loss(self, real_image, same_image):
        loss = tf.reduce_mean(tf.abs(real_image - same_image))
        return self.cycle_weight * self.identity_weight * loss

    def _perceptual_loss(self, disc_real_x, disc_real_y, disc_fake_x, disc_fake_y):
        loss = tf.reduce_mean(tf.square(disc_real_x - disc_fake_x))
        loss += tf.reduce_mean(tf.square(disc_real_y - disc_fake_y))
        return loss

    def training_step(self, record):
        x_batch, y_batch = record['x_batch'], record['y_batch']
        if callable(self.preprocessing):
            x_batch = self.preprocessing(x_batch)
            y_batch = self.preprocessing(y_batch)

        fake_Y = self.generator_XY(x_batch)
        cycled_X = self.generator_YX(fake_Y)

        fake_X = self.generator_YX(y_batch)
        cycled_Y = self.generator_XY(fake_X)

        same_X = self.generator_YX(x_batch)
        same_Y = self.generator_XY(y_batch)

        disc_real_x = self.discriminator_X(x_batch)
        disc_real_y = self.discriminator_Y(y_batch)

        disc_fake_x = self.discriminator_X(fake_X)
        disc_fake_y = self.discriminator_Y(fake_Y)

        loss_results = {}
        gen_XY_loss = self.losses[1](disc_fake_y)
        gen_YX_loss = self.losses[1](disc_fake_x)

        loss_results["discriminator_X"] = self.losses[0](
            disc_real_x, disc_fake_x)
        loss_results["discriminator_Y"] = self.losses[0](
            disc_real_y, disc_fake_y)

        loss_results["generator_XY"] = gen_XY_loss
        loss_results["generator_YX"] = gen_YX_loss

        loss_results["total_cycle"] = self.losses[2](
            x_batch, cycled_X) + self.losses[3](y_batch, cycled_Y)

        if self.use_perceptual_loss:
            loss_results['perceptual'] = self._perceptual_loss(
                disc_real_x, disc_real_y, disc_fake_x, disc_fake_y)

            loss_results["total_cycle"] += loss_results['perceptual']

        loss_results["total_generator_XY"] = gen_XY_loss + \
            loss_results["total_cycle"] + self.losses[4](y_batch, same_Y)

        loss_results["total_generator_YX"] = gen_YX_loss + \
            loss_results["total_cycle"] + self.losses[5](x_batch, same_X)

        return loss_results, self.model([x_batch, y_batch])

    def validation_step(self, record):
        pass

    def tape_step(self, tape, loss_values):
        if not self.optimizer:
            raise ValueError(
                "Optimizer not implemented in Model {}".format(self.__class__))

        generator_XY_gradients = tape.gradient(
            loss_values["total_generator_XY"],
            self.generator_XY.trainable_weights)

        generator_YX_gradients = tape.gradient(
            loss_values["total_generator_YX"],
            self.generator_YX.trainable_weights)

        discriminator_X_gradients = tape.gradient(
            loss_values["discriminator_X"],
            self.discriminator_X.trainable_weights)

        discriminator_Y_gradients = tape.gradient(
            loss_values["discriminator_Y"],
            self.discriminator_Y.trainable_weights)

        self.optimizer[0].apply_gradients(
            zip(generator_XY_gradients, self.generator_XY.trainable_variables))

        self.optimizer[1].apply_gradients(
            zip(generator_YX_gradients, self.generator_YX.trainable_variables))

        self.optimizer[2].apply_gradients(
            zip(discriminator_X_gradients, self.discriminator_X.trainable_variables))

        self.optimizer[3].apply_gradients(
            zip(discriminator_Y_gradients, self.discriminator_Y.trainable_variables))

        return [generator_XY_gradients, generator_YX_gradients, discriminator_X_gradients, discriminator_Y_gradients]

    def init_checkpoint(self):
        if self.ckpt is None:
            self.ckpt = tf.train.Checkpoint(
                generator_XY=self.generator_XY,
                generator_YX=self.generator_YX,
                discriminator_X=self.discriminator_X,
                discriminator_Y=self.discriminator_Y,
                optimizer1=self.optimizer[0],
                optimizer2=self.optimizer[1],
                optimizer3=self.optimizer[2],
                optimizer4=self.optimizer[3],
                step=tf.Variable(1)
            )
