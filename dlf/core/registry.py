""" The task of the registry is to register complex objects
by an keyword/alias that you easily can build and instanciate
these objects with a single keyword. This allows it in a easy
manner to parse a yaml configuration file and use these values
to instanciate the available objects.

"""

import tensorflow as tf
from importlib import import_module
from dlf.core.preprocessing import PreprocessingMethod
from dlf.core.callback import Callback
from dlf.core.evaluator import Evaluator


FRAMEWORK_CALLBACKS = {}
FRAMEWORK_DATA_GENERATORS = {}
FRAMEWORK_LOSSES = {}
FRAMEWORK_METRICS = {}
FRAMEWORK_MODELS = {}
FRAMEWORK_PREPROCESSING_METHODS = {}
FRAMEWORK_EVALUATORS = {}
FRAMEWORK_ACTIVE_EXPERIMENT = None


def import_framework_modules(module_folder, package):
    """ Auto import of all files in module folder

    # Note
        This is necessary for the register_* decorator to work properly.

    # Args
        module_folder: str.path to folder where files to import are located
        package: str. module path e.g. dlf.metrics
    """
    # auto import all files and register metrics
    # Path(__file__).parent
    for module in module_folder.iterdir():
        if module.name == '__init__.py' or module.suffix != '.py':
            continue
        module = f'{package}.{module.stem}'
        import_module(module)


def register_preprocessing_method(*names):
    """Decorator to register a preprocessing object to the framework

    # Args
        *names: Tuple(str). List of aliases for this preprocessing object

    # Raises
        ValueError: If the parent of this method is not of type [PreprocessingMethod](/dlf/core/preprocessing)
    """
    def decorator(cls):
        if not issubclass(cls, PreprocessingMethod):
            raise ValueError("invalid base class for class {}".format(cls))

        for name in names:
            FRAMEWORK_PREPROCESSING_METHODS[name] = cls

        return cls
    return decorator


def register_metric(*names):
    """Decorator to register a custom metric to the framework

    # Args
        *names: Tuple(str). List of aliases for this metric

    # Raises
        ValueError: If the parent of this method is not of type `tf.keras.metrics.Metrics`
        ValueError: If a given alias is not valid
    """
    def decorator(cls):
        if not issubclass(cls, tf.keras.metrics.Metric):
            raise ValueError("invalid base class for class {}".format(cls))

        FRAMEWORK_METRICS[cls.__name__] = cls  # alias
        for name in names:
            if not isinstance(name, str):
                raise ValueError(
                    "Invalid type of name '{}' for register_metric decorator".format(name))

            FRAMEWORK_METRICS[name] = cls
        return cls
    return decorator


def register_loss(*names):
    """Decorator to register a custom loss to the framework

    # Args
        *names: Tuple(str) List of aliases for this loss

    # Raises
        Exception: If object is not subclass of `tf.keras.losses.Loss`
        ValueError: If a given alias is not valid
    """
    def decorator(cls):
        if not issubclass(cls, tf.keras.losses.Loss):
            raise Exception("invalid base class for class {}".format(cls))

        FRAMEWORK_LOSSES[cls.__name__] = cls  # alias
        for name in names:
            if not isinstance(name, str):
                raise ValueError(
                    "Invalid type of  name '{}' for register_loss decorator".format(name))

            FRAMEWORK_LOSSES[name] = cls
        return cls
    return decorator


def register_data_generator(*names):
    """Decorator to register a data reader to the framework

    # Args
        *names: Tuple(str). List of aliases for this data reader

    # Raises
        ValueError: If a given alias is not valid
    """
    def decorator(cls):
        for name in names:
            if not isinstance(name, str):
                raise ValueError(
                    "Invalid type of  name '{}' for register_data_generator decorator".format(name))

            FRAMEWORK_DATA_GENERATORS[name] = cls
        return cls
    return decorator


def register_model(*names):
    """Decorator to register a custom model to the framework

    # Args
        *names: Tuple(str). List of aliases for this model

    # Raises
        ValueError: If a given alias is not valid
    """
    def decorator(cls):
        for name in names:
            if not isinstance(name, str):
                raise ValueError(
                    "Invalid type of name '{}' for register_model decorator".format(name))

            FRAMEWORK_MODELS[name] = cls

        return cls
    return decorator


def register_callback(*names):
    """Decorator to register a callback to the framework

    # Args
        *names: Tuple(str). List of aliases for this callback

    # Raises
        ValueError: If a given alias is not valid
    """
    def decorator(cls):
        for name in names:
            if not issubclass(cls, Callback):
                raise ValueError(
                    "Invalid type of name '{}' for register_callback decorator".format(name))

            FRAMEWORK_CALLBACKS[name] = cls

        return cls
    return decorator


def register_evaluator(*names):
    """Decorator to register an evaluator to the framework

    # Args
        *names: Tuple(str). List of aliases for this evaluator

    # Raises
        ValueError: If a given alias is not valid
    """
    def decorator(cls):
        for name in names:
            if not issubclass(cls, Evaluator):
                raise ValueError(
                    "Invalid type of name '{}' for register_evaluator decorator".format(name))

            FRAMEWORK_EVALUATORS[name] = cls

        return cls
    return decorator


def set_active_experiment(exp):
    """Sets active experiment to global state and
    allows all modules to access it

    # Arguments
        exp: dlf.core.Experiment. Active experiment
    """
    global FRAMEWORK_ACTIVE_EXPERIMENT
    FRAMEWORK_ACTIVE_EXPERIMENT = exp


def get_active_experiment():
    """Gets the current, active, experiment

    # Returns
        dlf.core.Experiment. Active experiment
    """
    global FRAMEWORK_ACTIVE_EXPERIMENT
    return FRAMEWORK_ACTIVE_EXPERIMENT
