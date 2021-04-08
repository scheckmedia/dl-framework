from tensorflow.keras.models import Model
from dlf.core.registry import register_model
from dlf.models.utils import get_output_layers
from dlf.core.builder import build_model_wrapper
from dlf.core.model import ModelWrapper


@register_model('feature_extractor', 'FeatureExtractor')
class FeatureExtractor(ModelWrapper):
    """ Feature Extractor

    This implementation allows it to initialize
    [pre-trained Keras models](https://www.tensorflow.org/api_docs/python/tf/keras/applications)
    for the usage as feature extractor.

    # Aliases
        - feature_extractor
        - FeatureExtractor

    # Arguments
        model_name: str. Name of the model e.g. ResNet50
        input_shape: tuple. Tuple containing the dimension of the input
        weights: str. path to weights or just 'imagenet'. Defaults to 'imagenet'.
        freeze: bool. If false the feature extractor is not trainable. Defaults to False.
        last_layer: str. Name of the last layer, used as feature extractor. Defaults to None.

    # Returns
        A Keras model instance.
    """

    def __init__(self, model_name, input_shape, weights='imagenet', last_layer=None, freeze=False, **kwargs):
        kwargs.update({"include_top": False,
                       "input_shape": input_shape, "weights": weights})

        feature_extractor = build_model_wrapper(model_name, kwargs)

        if last_layer is not None:
            layer, _ = get_output_layers(feature_extractor, last_layer)
            feature_extractor = Model(feature_extractor.input, layer)

        feature_extractor.model.trainable = not freeze

        super().__init__(feature_extractor.model, feature_extractor.preprocessing)
