""" Module for training
"""

import time
import os

import tensorflow as tf


def train():
    max_steps_train = 3000
    batch_size = 256
    model_dir = os.path.join('jobs/', str(int(time.time())))
    print('Model Directory:', model_dir)

    params = {
        'l2_scale': 1e-4,
        'lr_values': [1e-1, 1e-2, 1e-3],
        'lr_boundaries': [max_steps_train // 3, max_steps_train // 3 * 2],
        'bn_momentum': 0.99,
        'momentum': 0.9,
    }
    print('Hyperparameters:', params)

    config = tf.estimator.RunConfig(
        save_checkpoints_steps=1000,
        save_checkpoints_secs=None,
        save_summary_steps=100)

    tf.logging.set_verbosity(tf.logging.INFO)

    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=model_dir,
        config=config,
        params=params)

    filenames_train = [
        os.path.join(data_dir, 'data_batch_{}.bin'.format(i))
        for i in xrange(1, 6)]
    filenames_eval = [os.path.join(data_dir, 'test_batch.bin')]

    input_fn_train = get_input_fn(
        filenames=filenames_train,
        batch_size=batch_size,
        num_epochs=None,
        shuffle=True,
        augment=True)
    input_fn_eval = get_input_fn(
        filenames=filenames_eval,
        batch_size=batch_size,
        num_epochs=1,
        shuffle=False,
        augment=False)

    train_spec = tf.estimator.TrainSpec(
        input_fn=input_fn_train,
        max_steps=max_steps_train)
    eval_spec = tf.estimator.EvalSpec(
        input_fn=input_fn_eval,
        start_delay_secs=0,
        throttle_secs=60 * 5)

    tf.estimator.train_and_evaluate(
        train_spec=train_spec,
        eval_spec=eval_spec,
        estimator=estimator)
