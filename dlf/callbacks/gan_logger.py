import numpy as np
from dlf.core.experiment import ExperimentTarget
from dlf.core.registry import register_callback
from dlf.callbacks.tensorboard_logger import TensorboardLogger
from dlf.utils.visualization import visualize_cycle_gan


@register_callback('cycle_gan_logger', 'CycleGanLogger')
class CycleGanLogger(TensorboardLogger):
    """ A Callback for a Tensorboard visualization for a CycleGAN

    # Aliases
        - cycle_gan_logger
        - CycleGanLogger

    # Arguments
        num_visualizations: int, optional. Number of images to visualize. Defaults to 50.

    # YAML Configuration
    ```yaml
    callbacks:
        CycleGanLogger:
            num_visualizations: 200
    ```

    # Example

    ![](/computer-vision/dl-framework/img/callbacks/CycleGanLogger.gif)
    """

    def __init__(self, num_visualizations=50):
        super().__init__()

        self.num_visualizations = num_visualizations
        self._batch_counter = {}
        for target in ExperimentTarget:
            target = ExperimentTarget[target.name]
            self._batch_counter[target] = 0

        self.lhs = None
        self.rhs = None
        self._batch_size = 0

    def on_batch(self, logits, record, step, target):
        super().on_batch(logits, record, step, target)

        if target not in self._batch_counter or self._batch_counter[target] >= self.num_visualizations:
            return

        lhs_batch = record['x_batch'].numpy()
        rhs_batch = record['y_batch'].numpy()

        if not isinstance(self.lhs, np.ndarray):
            self.lhs = lhs_batch
            self.rhs = rhs_batch
        else:
            self.lhs = np.vstack([lhs_batch, self.lhs])
            self.rhs = np.vstack([rhs_batch, self.rhs])

        self._batch_counter[target] += lhs_batch.shape[0]

        if self._batch_size == 0:
            self._batch_size = lhs_batch.shape[0]

        drop = self._batch_counter[target] - self.num_visualizations
        if drop > 0:
            self.lhs = self.lhs[:-drop, :, :, :]
            self.rhs = self.rhs[:-drop, :, :, :]

    def on_evaluation(self, step, losses, metrics, target):
        super().on_evaluation(step, losses, metrics, target)

        wrapper = self.experiment.model_wrapper

        for idx in range(0, self.lhs.shape[0], self._batch_size):
            pos = idx * self._batch_size
            sub = pos + self._batch_size
            X = self.lhs[pos: sub, :, :, :]
            Y = self.rhs[pos: sub, :, :, :]

            fake_B = wrapper.generator_XY(X).numpy()
            fake_A = wrapper.generator_YX(Y).numpy()

            reconstr_A = wrapper.generator_YX(fake_B).numpy()
            reconstr_B = wrapper.generator_XY(fake_A).numpy()

            for subidx in range(0, X.shape[0]):
                imgs = [X[subidx, :, :, :], fake_B[subidx, :, :, :], reconstr_A[subidx, :, :, :],
                        Y[subidx, :, :, :], fake_A[subidx, :, :, :], reconstr_B[subidx, :, :, :]]

                plot_img = visualize_cycle_gan(
                    imgs, ['Original', 'Translated', 'Reconstructed'])

                name = 'sample {}'.format(idx + subidx)
                super().log_image(name, plot_img, step, target)
