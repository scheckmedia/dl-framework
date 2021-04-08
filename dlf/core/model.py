""" A ModelWrapper contains all model releated operations.
For instance the ModelWrapper contains the model which should be
trained or evaluated. Additional each single model requires in some
cases custom preprocessing methods like normalizations. This is also
part of the ModelWrapper. However, the required actions for a training
or a validation is abstracted through this wrapper.
"""
from tensorflow.keras.applications.imagenet_utils import preprocess_input as keras_preprocess_input
import tensorflow as tf


class ModelWrapper:
    """Base class which wraps all necessary objects and methods to make training feasible

    # Args
        model: tf.keras.models.Model. The model which should be wrapped
        preprocessing: callable, optional. A preprocessing function which will be executed. Defaults to None.
        optimizer: tf.keras.optimizers.Optimizer, optional. An optimizer used for training. Defaults to None.
        loss: dict[str,dict[str, Any]], optional. A loss function which should be build to use for training. Defaults to None.

    """

    def __init__(self, model, preprocessing=None, optimizer=None, loss=None, model_weights=None, additional_models=[], is_finetune=False):
        """Initializes a ModelWrapper
        """
        from dlf.core.builder import build_loss, build_optimizer
        self.ckpt = None
        self.model = model  # main model
        self.is_finetune = is_finetune
        self.__additional_models = additional_models

        self.preprocessing = preprocessing
        if preprocessing is None:
            self.preprocessing = self.preprocess_input_wrapper

        self.losses = []
        if loss is not None:
            self.loss = []
            for method, args in loss.items():
                args = {} if args is None else args
                self.losses += [build_loss(method, args)]

        if optimizer is not None:
            self.optimizer = []
            for opt in optimizer:
                method, args = list(opt.items())[0]
                args = {} if args is None else args
                self.optimizer += [build_optimizer(method, args)]

        if model_weights is not None:
            self.restore_model(model_weights)

    def preprocess_input_wrapper(self, x):
        """Model specific prerprocessing function

        This methods normalizes an input image in range -1 to 1 and
        is executedbefore it is feeds into a model

        # Args
            x: tf.Tensor. Image which should be normalized

        # Returns
            tf.Tensor: normalized image
        """
        return keras_preprocess_input(x, mode='tf')

    def training_step(self, record):
        """Executet during training for each batch of data

        This method is executed during training for each batch which should
        be feed into the network to train it.

        # Args
            record: dict. Training related data which contain the key 'x_batch' and 'y_batch'

        # Raises
            ValueError: If the ModelWrapper contains no loss function

        # Returns
            tuple(dict[str,Any], tf.Tensor): A dictionary with all assigned losses and the predicted logits
        """
        x_batch, y_batch = record['x_batch'], record['y_batch']
        if callable(self.preprocessing):
            x_batch = self.preprocessing(x_batch)

        logits = self.model(x_batch)

        if not self.losses:
            raise ValueError(
                "Loss not implemented in Model {}".format(self.__class__))

        losses = {}
        for loss in self.losses:
            losses[loss.name] = loss(y_batch, logits)

        return losses, logits

    def validation_step(self, record):
        """Exectued during validation

        See training_step

        # Args
            record: dict. Training related data which contain the key 'x_batch' and 'y_batch'

        # Raises
            ValueError: If the ModelWrapper contains no loss function

        # Returns
            tuple(dict[str,Any], tf.Tensor): A dictionary with all assigned losses and the predicted logits
        """
        return self.training_step(record)

    def tape_step(self, tape, loss_values):
        """Executed during training to backpropagate loss values

        # Args
            tape: Instance of GradientTape
            loss_values: dict[str, List[tf.Tensor]]. Dictionary containing the name of a loss and the corresponding values

        # Raises
            ValueError: If no optimizer is available

        # Returns
            List[tf.Tensor]: List of gradients
        """
        if not self.optimizer:
            raise ValueError(
                "Optimizer not implemented in Model {}".format(self.__class__))

        grads = tape.gradient(
            list(loss_values.values()), self.model.trainable_weights)
        self.optimizer[0].apply_gradients(
            zip(grads, self.model.trainable_weights))

        return grads

    def init_checkpoint(self, skip_optimizer=False):
        """Initialize a checkpoint object for saving and restoring
        """
        if not self.ckpt:
            kwargs = {
                "step": tf.Variable(1),
                "optimizer": self.optimizer[0],
                "model": self.model
            }

            if skip_optimizer:
                del kwargs['optimizer']

            self.ckpt = tf.train.Checkpoint(**kwargs)

    def save_model(self, path, step):
        """Methods which handles a request to save a model

        # Args
            path: str. Path where the checkpoints should be stored
            step: int. Current step
        """
        self.init_checkpoint()
        self.ckpt.step.assign(step)
        self.ckpt.save(path)

    def restore_model(self, path, step=None):
        """Restores a checkpoint at a given path

        # Arguments
            path: str. Path where the checkpoints are stored to restore
            step: int, optional. Sepcifies a checkpoint at step X which should be restored. Defaults to None.
        """

        self.init_checkpoint(self.is_finetune)
        if self.is_finetune: # if finetune it's ok if some layers are not restored
            # TODO: find a solution for this ugly work around

            try:
                available_variables = tf.train.list_variables(path)
                state = self.ckpt.restore(path)
                state.expect_partial()
            except:
                pass
        else:
            self.ckpt.restore(path).assert_consumed()

        tf.get_logger().info('restore checkpoint "{}" successful'.format(path))

    def num_iterations(self):
        """Get current number of iterations done by the optimizer

        # Returns
            int. Number of iterations
        """
        if len(self.optimizer) == 0:
            return 0

        return self.optimizer[0].iterations.numpy()

    @property
    def additional_models(self):
        """A list of additional models besides the main model

        # Returns
            List of tf.keras.models.Model. List of additional models
        """
        return self.__additional_models
