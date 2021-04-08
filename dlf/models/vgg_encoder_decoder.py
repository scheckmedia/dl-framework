from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Add, UpSampling2D, Conv2D, BatchNormalization, Activation, Conv2DTranspose
from dlf.core.registry import register_model
from dlf.core.model import ModelWrapper
from dlf.models.utils import get_output_layers


@register_model('VggEncoderDecoder', 'vgg_encoder_decoder')
class VggEncoderDecoder(ModelWrapper):
    """A SegNet like implementation of an Encoder-Decoder using VGG16.

    Compared to SegNet this implementation differs by using a 2D UpSampling layer (bilinear or nearest).
    Optional you can use the skip connections

    SegNet: [https://arxiv.org/abs/1511.00561](https://arxiv.org/abs/1511.00561)

    # Aliases:
        - VggEncoderDecoder
        - vgg_encoder_decoder

    # Args:
        input_shape: tuple(int, int , int). Input shape of this network
        num_classes: int. Number of classes
        use_skip_layers: bool, optional. If true the network uses skip layers. Defaults to True.
        use_transposed_conv: bool, optional. If true a transposed convolution instead of UpSample2D is used. Defaults to True.
        model_weights: str, optional. Path to the pretrained model weights. Defaults to None.
        summary: bool, optional. If true a summary of the model will be printed to stdout. Defaults to False.
        optimizer: list of dict, optional. Name of optimizer used for training.
        loss: list of dict, optional. List of loss objects to build for this model. Defaults to None.

    # Returns
        A Keras model instance.

    # YAML Configuration
        ```yaml
        model:
            vgg_encoder_decoder:
                input_shape:
                    - 512
                    - 512
                    - 3
                num_classes: &num_classes 7
                use_skip_layers: True
                use_transposed_conv: True
                model_weights: None
                summary: True
                optimizer:
                    - Adam:
                        learning_rate: 0.00001
                loss:
                    - SparseCategoricalCrossentropyIgnore:
                        num_classes: *num_classes
                        from_logits: False
        ```
    """

    def __init__(self, input_shape, num_classes, use_skip_layers=True, use_transposed_conv=True,
                 model_weights=None, summary=False, optimizer=None, loss=None, **kwargs):

        self.use_transposed = use_transposed_conv

        encoder = VGG16(include_top=False,
                        input_shape=input_shape, weights=None)

        skip_layer = {
            'block1_pool': 'block9_relu3',
            'block2_pool': 'block8_relu3',
            'block3_pool': 'block7_relu3',
            'block4_pool': 'block6_relu3',
        }

        skip_layer_output, skip_layer_names = None, None

        if use_skip_layers:
            skip_layer_output, skip_layer_names = get_output_layers(
                encoder, skip_layer.keys())

        x = self.__build_decoder(
            encoder.output, num_classes, skip_layer, skip_layer_output)

        x = Activation('softmax', name='softmax')(x)
        model = Model(encoder.input, x)

        if not loss:
            loss = {'SparseCategoricalCrossentropyIgnore': {
                'num_classes': num_classes,
                'from_logits': False}
            }

        if summary:
            model.summary()

        super().__init__(model, None, optimizer, loss, model_weights, **kwargs)

    def __build_decoder(self, encoder, num_classes, interpolation, skip_layer=None, skip_layer_output=None):
        blocks = [[512, 512, True], [512, 256, True], [
            256, 128, True], [128, 64, False], [64, num_classes, False]]

        x = encoder
        for idx, block in enumerate(blocks):
            filter, last_filter, is_three_conv = block
            x = self.__block(x, filter, last_filter, skip_layer, skip_layer_output, is_three_conv, idx == len(blocks) - 1,
                             name='block%d' % (6 + idx))

        return x

    def __block(self, x, filter, last_filter, skip_layer=None, skip_layer_output=None, is_three_conv=True, is_last=False, name='block'):
        if self.use_transposed:
            x = Conv2DTranspose(
                filter, kernel_size=3, strides=(2, 2), padding='same', name="%s_transposed_conv" % name)(x)
        else:
            x = UpSampling2D(size=2, interpolation='bilinear',
                             name="%s_upsample" % name)(x)
            x = Conv2D(filter, 3, padding='same', name="%s_conv1" % name)(x)

        x = BatchNormalization(name="%s_bn1" % name)(x)
        x = Activation('relu', name="%s_relu1" % name)(x)

        if is_three_conv:
            x = Conv2D(filter, 3, padding='same', name="%s_conv2" % name)(x)
            x = BatchNormalization(name="%s_bn2" % name)(x)
            x = Activation('relu', name="%s_relu" % name)(x)

        x = Conv2D(last_filter, 3 if not is_last else 1,
                   padding='same' if not is_last else 'valid', name="%s_conv3" % name)(x)
        x = BatchNormalization(name="%s_bn3" % name)(x)
        if not is_last:
            layer_name = "%s_relu3" % name
            x = Activation('relu', name=layer_name)(x)

            if skip_layer_output is not None and layer_name in skip_layer.values():
                skip_idx = list(skip_layer.values()).index(layer_name)
                layer = skip_layer_output[skip_idx]
                x = Add(name="%s_add_skip" % name)(
                    [x, layer.output])
        return x
