"""Contains all core implementations of a callback."""


class Callback:
    """ Base class for a callback """
    experiment = None

    def set_experiment(self, experiment):
        """Sets a reference to the current experiment.

        # Args
            experiment: Experiment. Current experiment
        """
        self.experiment = experiment

    def on_train_begin(self):
        """Called at the beginning of a training."""
        pass

    def on_train_end(self):
        """Called at the end of a training."""
        pass

    def on_evaluation(self, step, losses, metrics, target):
        """Called at every N evaluation step.

        # Args
            step: int. Current step
            losses: dict[str, float]. A dictionary containing loss name as key and the loss value as float
            metrics: dict[str, tf.keras.metrics.Metric]. A list of all available metrics
            target: dlf.core.experiment.ExperimentTarget. Target of this callback
        """
        pass

    def on_batch(self, logits, record, step, target):
        pass

    def on_gradient(self, gradient, record, step):
        pass
