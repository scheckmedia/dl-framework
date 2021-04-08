from dlf.core.registry import FRAMEWORK_LOSSES
from tqdm import tqdm
import numpy as np
from dlf.core.experiment import Experiment, ExperimentTarget
import argparse
import tensorflow as tf


def run(config):
    exp = Experiment.build(config)
    batch_size = exp.training.batch_size
    ckpt_path = exp.output_folder / 'checkpoints'
    if not ckpt_path.exists():
        ckpt_path.mkdir()

    ckpt_path = str(ckpt_path / 'ckpt')
    # set batch size for all reader
    exp.input_reader.set_batch_size(batch_size)

    # if exp.enable_function:
    #     exp.model_wrapper.training_step = tf.function(
    #         exp.model_wrapper.training_step)
    #     exp.model_wrapper.validation_step = tf.function(
    #         exp.model_wrapper.validation_step)
    #     exp.model_wrapper.tape_step = tf.function(exp.model_wrapper.tape_step)

    metrics = {
        ExperimentTarget.TRAIN: [],
        ExperimentTarget.VALIDATION: [],
        ExperimentTarget.TEST: []
    }

    if exp.input_reader.training_reader:
        train_ds = iter(exp.input_reader.training_reader)
        metrics[ExperimentTarget.TRAIN] = exp.training.generate_metrics()

    if exp.input_reader.validation_reader:
        val_ds = exp.input_reader.validation_reader
        metrics[ExperimentTarget.VALIDATION] = exp.training.generate_metrics()

    if exp.input_reader.test_reader:
        metrics[ExperimentTarget.TEST] = exp.training.generate_metrics()

    for callback in exp.training.callbacks:
        callback.on_train_begin()

    training_loss = {}
    monitor_values = {}
    validation_loss = {}

    # should we move this to save_strategy?
    if exp.training.save_strategy is not None:
        metrics_and_losses = []
        for k, v in metrics.items():
            m = ["{}_{}".format(k.value, x.name) for x in v]
            metrics_and_losses.extend(m)

            for l in exp.model_wrapper.losses:
                m = '{}_loss_{}'.format(k.value, l.name)
                metrics_and_losses.append(m)

        exp.training.save_strategy.check(metrics_and_losses)

    iterations_done = exp.model_wrapper.num_iterations() + 1
    with tqdm(range(iterations_done, exp.training.num_steps + 1), total=exp.training.num_steps, position=0, desc="Training: ") as training_progress:
        training_progress.update(iterations_done)
        for step in training_progress:
            try:
                record_example = next(train_ds)
            except StopIteration:
                continue

            if exp.input_reader.training_reader:
                with tf.GradientTape(persistent=True) as tape:
                    loss_values, logits = exp.model_wrapper.training_step(
                        record_example)

                for metric in metrics[ExperimentTarget.TRAIN]:
                    result = metric(record_example['y_batch'], logits)
                    monitor_values[ExperimentTarget.TRAIN.value + '_' +
                                   metric.name] = float(result)

                gradient = exp.model_wrapper.tape_step(tape, loss_values)

                for callback in exp.training.callbacks:
                    callback.on_gradient(
                        gradient, record_example, step)

                for callback in exp.training.callbacks:
                    callback.on_batch(logits, record_example,
                                      step, ExperimentTarget.TRAIN)

                for loss, value in loss_values.items():
                    if loss not in training_loss:
                        training_loss[loss] = []

                    training_loss[loss] += [value]

                msg = "Training step: {:d} with loss: {:.3f}".format(
                    step, float(tf.reduce_sum(list(loss_values.values()))))
                training_progress.set_description(msg)

                if step % exp.training.eval_every_step != 0:
                    continue

                for loss_name, loss_result in training_loss.items():
                    key = '{}_loss_{}'.format(
                        ExperimentTarget.TRAIN.value, loss_name)
                    monitor_values[key] = float(tf.reduce_sum(loss_result))

                for callback in exp.training.callbacks:
                    callback.on_evaluation(
                        step, training_loss, metrics[ExperimentTarget.TRAIN], ExperimentTarget.TRAIN)

                training_loss.clear()

            if exp.input_reader.validation_reader:
                with tqdm(enumerate(val_ds, 1), position=1, desc="Validation: ", leave=False) as validation_progress:
                    for val_step, val_record_example in validation_progress:
                        loss_values, logits = exp.model_wrapper.validation_step(
                            val_record_example)

                        for callback in exp.training.callbacks:
                            callback.on_batch(logits, val_record_example,
                                              step, ExperimentTarget.VALIDATION)

                        for loss, value in loss_values.items():
                            if loss not in validation_loss:
                                validation_loss[loss] = []

                            validation_loss[loss] += [value]

                        msg = "Validation step: {:d} with loss: {:.3f}".format(
                            val_step, float(tf.reduce_sum(list(loss_values.values()))))
                        validation_progress.set_description(msg)

                        for metric in metrics[ExperimentTarget.VALIDATION]:
                            result = metric(
                                val_record_example['y_batch'], logits)

                            monitor_values[ExperimentTarget.VALIDATION.value + '_' +
                                           metric.name] = float(result)

                    for loss_name, loss_result in validation_loss.items():
                        key = '{}_loss_{}'.format(
                            ExperimentTarget.VALIDATION.value, loss_name)
                        monitor_values[key] = float(tf.reduce_sum(loss_result))

                    for callback in exp.training.callbacks:
                        callback.on_evaluation(
                            step, validation_loss, metrics[ExperimentTarget.VALIDATION], ExperimentTarget.VALIDATION)

                    validation_loss.clear()

            if exp.training.save_strategy is not None and exp.training.save_strategy.need_to_save(monitor_values):
                exp.model_wrapper.save_model(ckpt_path, step)
            else:
                exp.model_wrapper.save_model(ckpt_path, step)

            for metric in metrics[ExperimentTarget.VALIDATION]:
                metric.reset_states()

            for metric in metrics[ExperimentTarget.TRAIN]:
                metric.reset_states()

        for callback in exp.training.callbacks:
            callback.on_train_end()

    training_progress.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        required=True, help="path to experiment configuration file")
    args = parser.parse_args()
    run(args.config)
