import tensorflow as tf
from dlf.core.experiment import ExperimentTarget
from dlf.core.registry import register_callback
from dlf.callbacks.tensorboard_logger import TensorboardLogger
from dlf.utils.visualization import visualize_predictions
from dlf.utils.scores import calculate_scores


@register_callback('segmentation_logger', 'SegmentationLogger')
class SegmentationLogger(TensorboardLogger):
    """Tensorboard visualization for segmentation tasks

    # Aliases
        - segmentation_logger
        - SegmentationLogger

    # Arguments
        num_classes: int. Number of classes, required for confusion matrix
        num_visualizations: int, optional. Number of images to visualize. Defaults to 50.
        opacity: float. Opacity of segmentation overlay. Defaults to 0.5.
        log_train_images: bool. If true training images are visualized as well. Defaults to False.

    # YAML Configuration
    ```yaml
    callbacks:
        SegmentationLogger:
            num_classes: 7
            num_visualizations: 200
    ```

    # Example

    ![](/computer-vision/dl-framework/img/callbacks/SegmentationLogger.gif)
    """

    def __init__(self, num_classes, num_visualizations=50, opacity=0.5, log_train_images=False):
        super().__init__()
        self.num_classes = num_classes
        self.num_visualizations = num_visualizations
        self.opacity = opacity
        self._batch_counter = {}
        for target in ExperimentTarget:
            target = ExperimentTarget[target.name]
            if not log_train_images and target == ExperimentTarget.TRAIN:
                continue

            self._batch_counter[target] = 0

        self.cm = tf.zeros(
            (self.num_classes, self.num_classes), dtype=tf.dtypes.int32)

    def on_batch(self, logits, record, step, target):
        super().on_batch(logits, record, step, target)

        if target == ExperimentTarget.VALIDATION:
            self.update_confusion_matrix(record['y_batch'], logits)

        if target not in self._batch_counter or self._batch_counter[target] >= self.num_visualizations:
            return

        gt = record['y_batch'].numpy()
        img = record['x_batch'].numpy()
        logits = logits.numpy().argmax(-1)

        out = visualize_predictions(img, gt, logits, self.opacity)
        self._batch_counter[target] += out.shape[0]

        name = 'image {} to {}'.format(
            self._batch_counter[target] - out.shape[0], self._batch_counter[target])

        drop = self._batch_counter[target] - self.num_visualizations
        if drop > 0:
            out = out[:-drop]

        super().log_image(name, out, step, target)

    def on_evaluation(self, step, losses, metrics, target):

        self._batch_counter[target] = 0

        if target == ExperimentTarget.VALIDATION:
            cm = self.cm.numpy()
            scores = calculate_scores(cm)
            labels = self.experiment.input_reader.validation_labels
            name = 'confusion_matrix'

            for key, value in scores.items():
                for class_id, label in labels.items():
                    tag = '{}_class_{}'.format(key, label)

                    self.__log_score(
                        key, label, value[class_id], step, target)

                self.__log_score(
                    key, 'mean', value.mean(), step, target)

            super().log_confusion_matrix(name, cm, labels.values(), step, target)
            self.cm = tf.zeros(
                (self.num_classes, self.num_classes), dtype=tf.dtypes.int32)
        super().on_evaluation(step, losses, metrics, target)

    def __log_score(self, score_name, label, value, step, target: ExperimentTarget):
        if label not in self._writer:
            self._writer[label] = tf.summary.create_file_writer(
                self.logdir / target.value / label)

        with self._writer[label].as_default():
            name = "validation_scores/{}".format(score_name)
            tf.summary.scalar(name, data=value, step=step)

    def update_confusion_matrix(self, y_true, y_pred):
        y_pred = tf.argmax(y_pred, axis=-1)

        if y_pred.shape.ndims > 1:
            y_pred = tf.reshape(y_pred, [-1])

        if y_true.shape.ndims > 1:
            y_true = tf.reshape(y_true, [-1])

        # ignore all labels >= num_classes
        valid_indices = tf.where(y_true < self.num_classes)
        y_true = tf.squeeze(tf.gather(y_true, valid_indices))
        y_pred = tf.squeeze(tf.gather(y_pred, valid_indices))

        self.cm += tf.math.confusion_matrix(y_true, y_pred, self.num_classes)
