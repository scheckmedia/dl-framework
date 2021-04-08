import tensorflow as tf
from tensorflow.keras.layers import Add, Conv2D, Softmax, UpSampling2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.models import Model

from dlf.core.registry import register_model
from dlf.core.builder import build_model_wrapper
from dlf.models.utils import get_output_layers
from dlf.core.model import ModelWrapper


@register_model('fcn', 'FCN')
class FCN(ModelWrapper):
    """A implementation of a Fully Convolutional Network

    # Aliases
        - FCN
        - fcn

    # Arguments
        feature_extractor: dict. A feature extractor
        num_classes: int. Number of classes used for segmentation
        output_shape: tuple or list. Output dimension (width, height) of the FCN
        skip_layers: list of str, optional.
            A list of layer which should be used for FCN skip connections.
            Defaults to None.
        interpolation_bilinear: bool, optional.
            If true bilinear instead of nearest is used as upsampling method.
            Defaults to True.
        model_weights: str, optional. Path to the pretrained model weights. Defaults to None.
        summary: bool, optional. If true a summary of the model will be printed to stdout. Defaults to False.
        optimizer: list of dict, optional. Name of optimizer used for training.
        loss: list of dict, optional. List of loss objects to build for this model. Defaults to None.

    # YAML Configuration
        Example usage of a FCN model in an experiment configuration. This example initiaizes
        the ResNet50 from tf.keras.applications package as feature extractor and an input shape
        of 512x512x3. Three layer of ResNet50 where used as skip connection.

        ```yaml
        model:
            fcn:
                feature_extractor:
                    model_name: ResNet50
                    input_shape:
                        - 512
                        - 512
                        - 3
                output_shape:
                    - 512
                    - 512
                skip_layers:
                - conv3_block4_out
                - conv4_block6_out
                - conv5_block1_out
                interpolation_bilinear: True
                num_classes: 7
                summary: True
                optimizer:
                - Adam:
                    learning_rate: 0.0001
        ```
    """

    def __init__(self, feature_extractor, num_classes, output_shape,
                 skip_layers=None, interpolation_bilinear=True, model_weights=None,
                 summary=False, optimizer=None, loss=None, **kwargs):
        if not isinstance(feature_extractor, dict):
            raise Exception("invalid type of feature extractor")

        feature_extractor_wrapper = build_model_wrapper(
            'feature_extractor', feature_extractor)
        feature_extractor = feature_extractor_wrapper.model

        feature_extractor_output_shape = feature_extractor.output_shape[-3:-1]
        if output_shape is None:
            output_shape = feature_extractor.input_shape[-3:-1]

        if skip_layers is not None:
            skip_layer_output, skip_layer_names = get_output_layers(
                feature_extractor, skip_layers)

        upsample_total = tuple(map(lambda x, y: int(
            x / y), output_shape, feature_extractor_output_shape))
        output_conv = feature_extractor.output
        output_up = Conv2D(filters=num_classes, kernel_size=(
            1, 1), activation='relu', name="conv_last")(output_conv)

        interpolation = 'bilinear' if interpolation_bilinear else 'nearest'

        if 0 not in upsample_total:
            output_up = UpSampling2D(
                size=upsample_total,
                interpolation=interpolation,
                name="upsample_%s_total" % interpolation)(output_up)

        pool = [output_up]

        if skip_layers and len(skip_layers) > 0:
            for idx, (skip_layer, skip_layer_name) in enumerate(zip(skip_layer_output, skip_layer_names)):
                previous_layer = pool[-1]

                scale_factor = tuple(
                    map(lambda x, y: int(y / x),
                        self.get_output_shape(skip_layer),
                        self.get_output_shape(previous_layer))
                )

                if 0 in scale_factor:
                    # we need downsampling

                    pool_size = tuple(
                        map(lambda x, y: int(x / y),
                            self.get_output_shape(skip_layer),
                            self.get_output_shape(previous_layer)
                            ))
                    scale_factor = (1, 1)
                    alias = skip_layer_name.split('/')[0]
                    skip_layer = MaxPooling2D(
                        pool_size=pool_size,
                        name='%s/skip_downsample_%d' % (alias, idx))(skip_layer.output)

                alias = skip_layer_name.split('/')[0]
                x = Conv2D(
                    filters=num_classes, kernel_size=(1, 1),
                    activation='relu', name="skip_%s/conv_%d" % (alias, idx))(skip_layer.output)
                x = BatchNormalization(name="skip_%s/bn_%d" % (alias, idx))(x)
                if scale_factor != (1, 1):
                    x = UpSampling2D(size=scale_factor,
                                     interpolation=interpolation,
                                     name="skip_%s/upsample_%d" % (alias, idx))(x)

                merge = Add(name="skip_%s/add_with_%s" % (
                    alias, pool[-1].name.split('/')[0]))([x, previous_layer])
                pool.append(merge)

        act = Softmax(name="act_softmax")(pool[-1])
        model = Model(inputs=feature_extractor.input, outputs=act)

        if summary:
            model.summary()

        if not loss:
            loss = {'SparseCategoricalCrossentropyIgnore':
                    {
                        'num_classes': num_classes,
                        'from_logits': False
                    }
                    }

        super().__init__(model, feature_extractor_wrapper.preprocessing,
                         optimizer, loss, model_weights, **kwargs)

    def get_output_shape(self, obj):
        if isinstance(obj, tf.Tensor):
            shape = obj.shape[-3:-1]
        else:
            shape = obj.output_shape[-3:-1]
        return shape
