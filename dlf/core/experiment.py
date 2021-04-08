""" The experiment module contains all building processes for an experiment
based on a yaml configuration and provides an interface for interaction with
the required properties.
"""
import yaml
import sys
import operator
from enum import Enum
from pathlib import Path

import tensorflow as tf

from dlf.core.builder import build_metric, build_model_wrapper
from dlf.core.builder import build_data_generator, build_callback
from dlf.core.registry import set_active_experiment
import warnings


class Experiment:
    """A representation of an experiment

    This class represents a setup of an experiment. Based on *.yaml configuration
    file this class maps these values and initializes all required elements.

    # Args
        config: str. Path to a configuration file

    # Raises
        FileNotFoundError: If the specified configuration file does not exists
        KeyError: If required keys are missing in a configuration file
    """
    training = None

    @staticmethod
    def build(config):
        return Experiment(config)

    def __init__(self, config):
        """Initializer for an experiment
        """
        if not Path(config).exists():
            raise FileNotFoundError(f'no config file found at {config}')

        with open(config) as f:
            self.config = yaml.load(f, Loader=yaml.Loader)
            self.__parse()

    def __parse(self):
        self.__validate()
        set_active_experiment(self)

        self.output_folder = Path(self.config["experiment"]["output_folder"])
        tf.get_logger().setLevel(
            self.config["experiment"].get('log_level', 20))
        # self.enable_function = self.config["experiment"].get(
        #     "enable_function", True)

        if not self.output_folder.exists():
            self.output_folder.mkdir()

        w = ModelSettings(self.config["model"])
        self.model_wrapper = w.wrapper

        if 'training' in self.config:
            self.training = TrainingSettings(self.config["training"])

        self.input_reader = InputReaderSettings(self.config["input_reader"])

        for callback in self.training.callbacks:
            callback.set_experiment(self)

    def __validate(self):
        required_keys = set(["experiment", "model", "input_reader"])
        missing = required_keys - self.config.keys()

        if len(missing):
            raise KeyError(
                "Missing required keys in config file: {}".format(missing))

        if 'output_folder' not in self.config['experiment']:
            raise KeyError("Missing output folder for experiment results")


class InputReaderSettings:
    """Container for all selected input readers

    # Args
        settings: dict[str,Any]. Dictionary of `input_reader` section

    # YAML Configuration
    ```yaml
    input_reader:
        training_reader:
            name: fs_random_unpaired_reader
            path_lhs: /mnt/data/datasets/theodore_wheeled_walker/*_img.png
            path_rhs: /mnt/data/datasets/omnidetector-Flat/JPEGImages/*.jpg
            lhs_limit: 10000
            rhs_limit: 10000
            shuffle_buffer: 100
            preprocess_list:
            resize:
                output_shape:
                - 512
                - 512
    ```
    """
    _training_reader = None
    _training_labels = None
    _training_padded_batch_shape = None
    _validation_reader = None
    _validation_labels = None
    _validation_padded_batch_shape = None
    _test_reader = None
    _test_labels = None
    _test_padded_batch_shape = None

    def __init__(self, settings):
        """Intializer for an InputReaderSettings object
        """
        self.__settings = settings
        self.__parse()

    def __parse(self):
        valid = ['training_reader', 'validation_reader', 'test_reader']

        for reader, args in self.__settings.items():
            if reader not in valid:
                warnings.warn(
                    'skip reader "{}" becaus it\'s not a valid key'.format(reader))

            name = args["name"]
            args.pop("name")

            self.__settings[reader] = build_data_generator(name, args)

        if 'training_reader' in self.__settings and 'batch' in dir(self.__settings['training_reader'].dataset):
            self._training_reader = self.__settings['training_reader'].dataset
            self._training_labels = self.__settings['training_reader'].labels
            self._training_padded_batch_shape = self.__settings['training_reader'].padded_batch_shape

        if 'validation_reader' in self.__settings and 'batch' in dir(self.__settings['validation_reader'].dataset):
            self._validation_reader = self.__settings['validation_reader'].dataset
            self._validation_labels = self.__settings['validation_reader'].labels
            self._validation_padded_batch_shape = self.__settings[
                'validation_reader'].padded_batch_shape

        if 'test_reader' in self.__settings and 'batch' in dir(self.__settings['test_reader'].dataset):
            self._test_reader = self.__settings['test_reader'].dataset
            self._test_labels = self.__settings['test_reader'].labels
            self._test_padded_batch_shape = self.__settings['test_reader'].padded_batch_shape

    def set_batch_size(self, batch_size):
        """Updates the batch size of all input readers

        # Args
            batch_size: int. Size of a batch
        """
        if self._training_reader:
            if self._training_padded_batch_shape:
                self._training_reader = self._training_reader.repeat(
                    count=-1).padded_batch(batch_size, self._training_padded_batch_shape,
                                           self.__settings['training_reader'].padded_values,
                                           drop_remainder=True)
            else:
                self._training_reader = self._training_reader.repeat(
                    count=-1).batch(batch_size, drop_remainder=True)

        if self._validation_reader:
            if self._validation_padded_batch_shape:
                self._validation_reader = self._validation_reader.padded_batch(
                    batch_size, self._validation_padded_batch_shape,
                    self.__settings['validation_reader'].padded_values,
                    drop_remainder=True)
            else:
                self._validation_reader = self._validation_reader.batch(
                    batch_size, drop_remainder=True)

        if self._test_reader:
            if self._test_padded_batch_shape:
                self._test_reader = self._test_reader.padded_batch(
                    batch_size, self._test_padded_batch_shape,
                    self.__settings['test_reader'].padded_values,
                    drop_remainder=True)
            else:
                self._test_reader = self._test_reader.batch(
                    batch_size, drop_remainder=True)

    @property
    def training_reader(self):
        """Instance of an active training reader

        # Returns
            - tf.data.Dataset: If available the dataset instance will be returned
            - None: If there is no training_reader specified, None will be the return value
        """
        return self._training_reader

    @property
    def validation_reader(self):
        """Instance of an active validation reader

        # Returns
            - tf.data.Dataset: If available the dataset instance will be returned
            - None: If there is no validation_reader specified, None will be the return value
        """
        return self._validation_reader

    @property
    def test_reader(self):
        """Instance of an active test reader

        # Returns
            - tf.data.Dataset: If available the dataset instance will be returned
            - None: If there is no test_reader specified, None will be the return value
        """
        return self._test_reader

    @property
    def training_labels(self):
        """List of all provided labels of training_reader

        # Returns
            - dict[0,str]: Dictionary with category id as key and label as value
            - None: If no labels are available
        """
        return self._training_labels

    @property
    def validation_labels(self):
        """List of all provided labels of validation_reader

        # Returns
            - dict[0,str]: Dictionary with category id as key and label as value
            - None: If no labels are available
        """
        return self._validation_labels

    @property
    def test_labels(self):
        """List of all provided labels of test_reader

        # Returns
            - dict[0,str]: Dictionary with category id as key and label as value
            - None: If no labels are available
        """
        return self._test_labels


class ModelSettings:
    """Object to describe model specific settings

    # Args
        settings: dict[str,Any]. Dictionary to initialize a model

    # Raises
        KeyError: If a specified model is not available or the setup is wrong
    """
    wrapper = None

    def __init__(self, settings):
        """Initializes a model
        """
        self._settings = settings

        if len(settings.keys()) != 1:
            raise KeyError(
                "model key can only contain a single entry with a valid model")

        self.__build()

    def __build(self):
        model = list(self._settings.keys())[0]
        kwargs = list(self._settings.values())[0]

        self.wrapper = build_model_wrapper(model, kwargs)


class TrainingSettings:
    """Object to describe the selected training settings

    This object parses the `training` section of a config file.
    Based on the configuration callbacks and metrics are initialized
    for the experiment.

    # Args
        settings: dict[key, Any]. The training section of configuration file

    # Raises
        Exception: If the monitor value is not specified
        KeyError: If the configuration files contains not all required keys for the training section
    """
    num_steps = None
    batch_size = None
    save_strategy = None
    callbacks = []

    def __init__(self, settings):
        """Initializer for TrainingSettings
        """
        self._settings = settings
        self.__parse()

    def generate_metrics(self):
        """Generate instances of the selected and available metrics

        # Returns
            A List of instances of selected metrics
        """
        metrics = []
        if "metrics" not in self._settings or not isinstance(self._settings["metrics"], dict):
            return metrics

        for method, args in self._settings['metrics'].items():
            args = {} if args is None else args
            metrics += [build_metric(method, args)]
        return metrics

    def __parse(self):
        self.__validate()
        self.__build_callbacks()
        self.__build_save_strategy()

        self.num_steps = self._settings["num_steps"]
        self.batch_size = self._settings.get("batch_size", 4)
        self.eval_every_step = self._settings.get("eval_every_step", 1000)

    def __build_save_strategy(self):
        self.save_strategy = self._settings.get("save_strategy", None)
        if self.save_strategy is None:
            return

        if 'monitor' not in self.save_strategy.keys():
            raise Exception("Save strategy must contain a value to monitor")

        mode = self.save_strategy.get('mode', 'max')
        monitor = self.save_strategy['monitor']
        self.save_strategy = SaveStrategy(monitor, mode)

    def __build_callbacks(self):
        if "callbacks" not in self._settings or not isinstance(self._settings["callbacks"], dict):
            return

        for callback, args in self._settings["callbacks"].items():
            args = {} if args is None else args
            self.callbacks += [build_callback(callback, args)]

    def __validate(self):
        required_keys = set(
            ["num_steps", "batch_size"]
        )
        missing = required_keys - self._settings.keys()

        if len(missing):
            raise KeyError(
                "Missing required keys for training in config file: {}".format(missing))


class SaveStrategy:
    """Represents a saving strategy for a model during training

    This class implements a simple saving strategy to save models during training.
    With the configuration file you have the opportunity to specify different metrics and losses.
    Each of them have an unique name and regarding the evaluation target ... an alias will be prepended.

    For instance you setup the metric SparseMeanIOU. The value of this metric will be propagated through
    all evaluation steps. For training the monitor value will be `train_sparse_mean_iou` and for validation
    `val_sparse_mean_iou`. As second parameter you have to specify whether to save the model if the min or
    max value of this metric improves during training.

    # Args
        monitor: str. Value to monitor during training
        mode: {'max','min'}. Whether to save the model if the minimum value or maximum value improves
    """

    def __init__(self, monitor, mode):
        """Initializes the training strategy
        """
        self.__monitor = monitor
        self.__mode = mode

        if mode == 'max':
            self.__current_value = sys.float_info.min
            self.__operator = operator.gt
        else:
            self.__current_value = sys.float_info.max
            self.__operator = operator.lt

    @property
    def monitor(self):
        """Value to watch for improvements

        # Returns
            str: Active value to watch in monitor values to detect improvements
        """
        return self.__monitor

    @property
    def mode(self):
        """Current mode for the saving strategy

        # Returns
            {'min','max'}: [description]
        """
        return self.__mode

    def need_to_save(self, monitor_values):
        """Method which decides whether a model should be saved or not

        # Args
            monitor_values: dict[str,float]. Dictionary containing monitor values

        # Returns
            bool: Describes whether a model has been improved and needs to be saved or not.
        """
        if self.__monitor not in monitor_values:
            warnings.warn(
                'Monitor value "{}" is not available'.format(self.__monitor))
            return

        value = monitor_values[self.__monitor]
        res = self.__operator(value, self.__current_value)
        if res:
            self.__current_value = value
        return res

    def check(self, monitor_values):
        if self.__monitor not in monitor_values:
            message = 'Monitor value "{}" not found. List of available values below:\n'.format(
                self.__monitor)
            for m in monitor_values:
                message += "{}\n".format(m)

            raise ValueError(message)


class ExperimentTarget(Enum):
    """ Enum to identify the current state during the experiment"""

    TRAIN = "train"
    """specifies that the target is training"""

    VALIDATION = "validation"
    """specifies that the target is validation"""

    TEST = "test"
    """specifies that the target is test"""
