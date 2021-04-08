import tensorflow as tf
import numpy as np

from dlf.core.experiment import ExperimentTarget
from dlf.core.registry import register_callback, get_active_experiment
from dlf.callbacks.tensorboard_logger import TensorboardLogger
from dlf.utils.visualization import visualize_object_detection
from dlf.utils.box_utils import compute_area
from dlf.core.builder import build_evaluator


@register_callback('object_detection_logger', 'ObjectDetectionLogger')
class ObjectDetectionLogger(TensorboardLogger):
    """Tensorboard visualization for object detection tasks

    # Aliases
        - object_detection_logger
        - ObjectDetectionLogger

    # Arguments
        evaluator: dlf.core.evaluator. Evaluator used for the summerie and metrics
        num_visualizations: int, optional. Number of images to visualize. Defaults to 50.
        log_train_images: bool. If true training images are visualized as well. Defaults to False.

    # YAML Configuration
    ```yaml
    callbacks:
        ObjectDetectionLogger:
            num_visualizations: 200
    ```

    # Example

    ![](/computer-vision/dl-framework/img/callbacks/ObjectDetectionLogger.gif)
    """

    def __init__(self, evaluator, num_visualizations=50, log_train_images=False):
        super().__init__()

        self.evaluators = {}
        self.num_visualizations = num_visualizations

        self._batch_counter = {}
        for target in ExperimentTarget:
            target = ExperimentTarget[target.name]
            if not log_train_images and target == ExperimentTarget.TRAIN:
                continue

            self._batch_counter[target] = 0

            evals = []
            for k, v in evaluator.items():
                evals += [build_evaluator(k, v)]

            self.evaluators[target] = evals

    def on_train_begin(self):
        self.labels = {
            ExperimentTarget.TRAIN: get_active_experiment().input_reader.training_labels,
            ExperimentTarget.VALIDATION: get_active_experiment().input_reader.validation_labels,
            ExperimentTarget.TEST: get_active_experiment().input_reader.test_labels
        }

        for target in ExperimentTarget:
            if target not in self.evaluators:
                continue

            for evaluator in self.evaluators[target]:
                evaluator.labels = self.labels[target]
        return super().on_train_begin()

    def on_batch(self, logits, record, step, target):
        super().on_batch(logits, record, step, target)

        pred_classes, pred_boxes, pred_scores = logits
        pred_scores = [x.numpy() for x in pred_scores]

        x_batch, y_batch = record['x_batch'], record['y_batch']
        image_shape = np.array(x_batch.shape[1:])

        gt_labels = []
        gt_boxes = []
        gt_areas = []
        for i in range(x_batch.shape[0]):
            valid_indices = y_batch['gt_labels'][i] != -1

            gt_label = tf.boolean_mask(
                y_batch['gt_labels'][i], valid_indices)
            gt_box = tf.boolean_mask(
                y_batch['gt_boxes'][i], valid_indices)

            gt_areas.append(compute_area(
                gt_box[..., :2], gt_box[..., 2:]).numpy())
            gt_labels.append(gt_label.numpy().astype(np.int32))
            gt_boxes.append(gt_box.numpy().astype(np.float32))

        gt_labels = np.array(gt_labels)
        gt_boxes = np.array(gt_boxes)
        gt_ids = y_batch['ids'].numpy()
        gt_areas = np.array(gt_areas)

        # update evaulation
        if target in self.evaluators:
            for evaluator in self.evaluators[target]:
                evaluator.add_batch(image_shape, np.array(pred_boxes), np.array(pred_classes), np.array(pred_scores),
                                    gt_boxes, gt_labels, gt_ids, gt_areas)

        if target not in self._batch_counter or self._batch_counter[target] >= self.num_visualizations:
            return

        out = visualize_object_detection(x_batch, pred_boxes, pred_classes, pred_scores,
                                         gt_boxes, gt_labels, classes=self.labels[target])
        self._batch_counter[target] += out.shape[0]

        name = 'image {} to {}'.format(
            self._batch_counter[target] - out.shape[0], self._batch_counter[target])

        drop = self._batch_counter[target] - self.num_visualizations
        if drop > 0:
            out = out[:-drop]

        super().log_image(name, out, step, target)

    def on_evaluation(self, step, losses, metrics, target):
        super().on_evaluation(step, losses, metrics, target)

        if target not in self._batch_counter:
            return

        self._batch_counter[target] = 0

        if target in self.evaluators:
            for evaluator in self.evaluators[target]:
                scores = evaluator.evaluate()
                for key, value in scores.items():
                    with self._writer[target].as_default():
                        tf.summary.scalar(key, data=value, step=step)
