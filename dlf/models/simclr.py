import tensorflow as tf
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.models import Model

from dlf.core.registry import register_model
from dlf.core.builder import build_model_wrapper
from dlf.models.utils import add_weight_decay
from dlf.core.model import ModelWrapper


@register_model('SimCLR', 'sim_clr', 'simclr')
class SimCLR(ModelWrapper):
    """A implementation of SimCLR

    # Aliases
        - SimCLR
        - sim_clr
        - simclr

    # Arguments
        feature_extractor: dict. A feature extractor
        num_projection_dimension: int. Number of projection dimensions for unsupervised approach
        linear_evaluation: dict. Defines linear evaulation options, if None linear evaluation is desabled. Defaults to None.
            - num_classes: int. Number of classes for linear evaluation
            - linear_train_at_step: int. Number at which step the linear evaluation starts
            - freeze_base_network: bool. If true, the feature extractor is not trainable during linear evaluation. Defaults to True.
        weight_decay: float. Amount of weight decay to use. Defaults to 1e-6.
        use_batch_norm: bool. If true, batch norm for the projection head is used. Defaults to True.
        model_weights: str, optional. Path to the pretrained model weights. Defaults to None.
        summary: bool, optional. If true a summary of the model will be printed to stdout. Defaults to False.
        optimizer: list of dict, optional. Name of optimizer used for training.
        loss: list of dict, optional. List of loss objects to build for this model. Defaults to None.

    # YAML Configuration
        ```yaml
        model:
            simclr:
                feature_extractor:
                    model_name: ResNet50
                    input_shape:
                        - &width 224
                        - &height 224
                        - 3
                num_projection_dimension: 128
                linear_evaluation:
                    num_classes: 62
                    linear_train_at_step: 5000
                    freeze_base_network: False
                num_classes: 62
                use_batch_norm: True
                linear_train_at_step: 5000
                weight_decay: 1e-6
                summary: True
                loss:
                    NTXentLoss:
                        batch_size: &batch_size 16
                        temperature: 0.5
                        use_cosine_similarity: True
                    # categorical_labels is True in reader otherwise it should be SparseCategoricalCrossentropy
                    CategoricalCrossentropy:
                        from_logits: True
                        reduction: "none"
                optimizer:
                    - Adam: # lr for simclr unsupervised
                        learning_rate: 0.0001
                    - SGD: # lr for linear evaluation
                        learning_rate: 0.001
                        momentum: 0.9
        ```

        # References
            - [A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/abs/2002.05709)
    """

    def __init__(self, feature_extractor, num_projection_dimension=128, linear_evaluation=None, weight_decay=1e-6,
                 use_batch_norm=True, model_weights=None, summary=False, optimizer=None, loss=None, **kwargs):

        if not isinstance(feature_extractor, dict):
            raise Exception("invalid type of feature extractor")

        feature_extractor_wrapper = build_model_wrapper(
            'feature_extractor', feature_extractor)
        feature_extractor = feature_extractor_wrapper.model

        out = feature_extractor.outputs[0]
        h = GlobalAveragePooling2D(name='AVG')(out)

        x = Dense(h.shape[-1], use_bias=True, activation='relu',
                  kernel_initializer=tf.keras.initializers.random_normal(
            stddev=.01),
            name='projection/fc1')(h)
        if use_batch_norm:
            x = BatchNormalization(name="projection/bn_fc1")(x)

        z = Dense(num_projection_dimension, use_bias=False,
                  kernel_initializer=tf.keras.initializers.random_normal(
                      stddev=.01),
                  activation='linear', name='projection/fc2')(x)

        if use_batch_norm:
            z = BatchNormalization(name="projection/bn_fc2")(z)

        self.simclr_model = Model(
            inputs=feature_extractor.input, outputs=[h, z], name="SimCLR Projection")
        add_weight_decay(self.simclr_model, weight_decay)

        additional_models = []

        self.linear_model = None
        self.linear_train_at_step = None
        if linear_evaluation is not None:
            if 'num_classes' not in linear_evaluation or 'linear_train_at_step' not in linear_evaluation:
                raise ValueError(
                    "Linear Evaluation dict requires num_classes and linear_train_at_step")

            self.linear_freeze_base = True
            if 'freeze_base_network' in linear_evaluation:
                self.linear_freeze_base = linear_evaluation['freeze_base_network']

            x = Dense(linear_evaluation['num_classes'],
                      kernel_initializer=tf.keras.initializers.random_normal(
                      stddev=.01),
                      name='linear_evaluation')(h)
            self.linear_model = Model(
                inputs=feature_extractor.input, outputs=[x], name="Linear Evaluation")

            add_weight_decay(self.linear_model, weight_decay)

            additional_models.append(self.linear_model)
            self.linear_train_at_step = linear_evaluation['linear_train_at_step']

        if summary:
            self.simclr_model.summary()
            if self.linear_model is not None:
                self.linear_model.summary()

        super().__init__(self.simclr_model, feature_extractor_wrapper.preprocessing,
                         optimizer, loss, model_weights, additional_models=additional_models, **kwargs)

        if len(self.optimizer) != 2:
            raise ValueError("SimCLR requires two optimizer!")

    def training_step(self, record):
        if ('x_batch' not in record and 'y_batch' not in record):
            raise ValueError("Invalid data reader for SimCLR network")

        if 'x1' not in record['x_batch'] or 'x2' not in record['x_batch']:
            raise ValueError(
                "Data reader output is not right for SimCLR, enable SimCLR mode")
        x1 = record['x_batch']['x1']
        x2 = record['x_batch']['x2']

        loss_values = {}
        if self.linear_train_at_step is None or self.optimizer[0].iterations < self.linear_train_at_step:
            if callable(self.preprocessing):
                x1 = self.preprocessing(x1)
                x2 = self.preprocessing(x2)

            his, zis = self.simclr_model(
                x1, training=True)
            hjs, zjs = self.simclr_model(
                x2, training=True)

            zis = tf.math.l2_normalize(zis, axis=1)
            zjs = tf.math.l2_normalize(zjs, axis=1)

            loss, logits = self.losses[0](zis, zjs)
            loss_values[self.losses[0].name] = loss
        else:
            if callable(self.preprocessing):
                x1 = self.preprocessing(x1)

            logits = self.linear_model(
                x1, training=True)

            loss = self.losses[1](y_pred=logits, y_true=record['y_batch'])
            loss_values[self.losses[1].name] = loss

        return loss_values, logits

    def tape_step(self, tape, loss_values):
        if self.linear_train_at_step is None or self.optimizer[0].iterations < self.linear_train_at_step:
            loss = loss_values[self.losses[0].name]
            grads = tape.gradient(loss, self.simclr_model.trainable_weights)
            self.optimizer[0].apply_gradients(
                zip(grads, self.simclr_model.trainable_weights))
        else:
            loss = loss_values[self.losses[1].name]

            if self.linear_freeze_base:
                trainable_weights = self.linear_model.trainable_weights[-2:]
            else:
                trainable_weights = self.linear_model.trainable_weights

            grads = tape.gradient(loss, trainable_weights)
            self.optimizer[1].apply_gradients(
                zip(grads, trainable_weights))

        return grads

    def init_checkpoint(self):
        if self.ckpt is None:
            kwargs = {
                'simclr_model': self.simclr_model,
                'simclr_optimizer': self.optimizer[0],
                'step': tf.Variable(1)
            }

            if self.linear_model is not None:
                kwargs['linear_evaluation'] = self.linear_model
                kwargs['linear_optimizer'] = self.optimizer[1]

            self.ckpt = tf.train.Checkpoint(**kwargs)

    def num_iterations(self):
        num = super().num_iterations()

        if self.linear_train_at_step is None or num < self.linear_train_at_step:
            num += self.optimizer[1].iterations.numpy()

        return num
