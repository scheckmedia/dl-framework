import tensorflow as tf
import numpy as np
from tensorflow.python.ops import summary_ops_v2
from tensorflow.python.keras import backend as K
from tensorflow.keras.optimizers.schedules import LearningRateSchedule
from dlf.core.experiment import ExperimentTarget
from dlf.core.callback import Callback
from dlf.core.registry import register_callback
from dlf.utils.visualization import visualize_confusion_matrix


@register_callback('tensorboard_logger', 'TensorboardLogger')
class TensorboardLogger(Callback):
    """A callback which logs every available metric to TensorBoard.

    # Note
        This should be the base class for every TensorBoard logging related implementation

    # Aliases
        - tensorboard_logger
        - TensorboardLogger

    ```yaml
    callbacks:
        TensorboardLogger:
    ```
    """

    def __init__(self):
        super().__init__()

    def set_experiment(self, experiment):
        super().set_experiment(experiment)
        self.logdir = self.experiment.output_folder
        self._model = self.experiment.model_wrapper.model
        self._writer = {}

        for target in ExperimentTarget:
            self._writer[ExperimentTarget[target.name]] = tf.summary.create_file_writer(
                self.logdir / target.value)

    def on_train_begin(self):
        self._write_graph()
        self._log_learning_rate(0, ExperimentTarget.TRAIN)

    def on_train_end(self):
        self._close_writer()

    def on_evaluation(self, step, losses, metrics, target: ExperimentTarget):
        for loss, value in losses.items():
            self.log_scalar('loss_{}'.format(loss),
                            np.mean(value), step, target)
        for metric in metrics:
            result = float(metric.result())
            self.log_scalar(metric.name, result, step, target)

        if target == ExperimentTarget.TRAIN:
            self._log_learning_rate(step, target)

    def log_scalar(self, tag, value, step, target: ExperimentTarget):
        with self._writer[target].as_default():
            name = "{}/{}".format(target.value, tag)
            tf.summary.scalar(name, data=value, step=step)

    def log_image(self, tag, image, step, target: ExperimentTarget):
        with self._writer[target].as_default():
            name = "{}/{}".format(target.value, tag)
            if len(image.shape) == 3:
                image = np.expand_dims(image, 0)
            tf.summary.image(name, image, step, max_outputs=256)

    def log_confusion_matrix(self, tag, cm, labels, step, target: ExperimentTarget):
        img = visualize_confusion_matrix(cm, labels, None)
        self.log_image(tag, img, step, target)

    def _write_graph(self):
        with self._writer[ExperimentTarget.TRAIN].as_default():
            modellist = [self._model]
            modellist.extend(self.experiment.model_wrapper.additional_models)

            for model in modellist:
                if not model.run_eagerly:
                    summary_ops_v2.graph(K.get_graph(), step=0)

                summary_writable = (
                    model._is_graph_network or  # pylint: disable=protected-access
                    model.__class__.__name__ == 'Sequential')  # pylint: disable=protected-access
                if summary_writable:
                    summary_ops_v2.keras_model('keras', model, step=0)

    def _close_writer(self):
        for writer in self._writer.values():
            writer.close()

        self._writer.clear()

    def _log_learning_rate(self, step, target):
        for idx, optimizer in enumerate(self.experiment.model_wrapper.optimizer):
            if not issubclass(optimizer.learning_rate.__class__, LearningRateSchedule):
                continue

            lr = optimizer._decayed_lr(tf.float32)

            with self._writer[target].as_default():
                name = "{}/learning_rate_optimizer_{}".format(
                    target.value, idx + 1)
                tf.summary.scalar(name, data=lr, step=step)
