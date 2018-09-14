""" Module for building model
"""

import tensorflow as tf

def batch_norm(inputs, momentum, training):
    return tf.layers.batch_normalization(
        inputs=inputs,
        scale=False,
        momentum=momentum,
        training=training,
        fused=True)

def resnet_block(inputs, filters, strides, training, params):
    kernel_initializer = tf.initializers.variance_scaling(scale=2.0)
    kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=params['l2_scale'])

    inputs = batch_norm(
        inputs=inputs,
        momentum=params['bn_momentum'],
        training=training)
    inputs = tf.nn.relu(inputs)

    hidden = tf.layers.conv2d(
        inputs=inputs,
        filters=filters,
        kernel_size=3,
        strides=strides,
        use_bias=False,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer,
        padding='same')

    hidden = batch_norm(
        inputs=hidden,
        momentum=params['bn_momentum'],
        training=training)
    hidden = tf.nn.relu(hidden)
    hidden = tf.layers.conv2d(
        inputs=hidden,
        filters=filters,
        kernel_size=3,
        strides=1,
        use_bias=False,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer,
        padding='same')

    inputs_padded = tf.layers.conv2d(
        inputs=inputs,
        filters=filters,
        kernel_size=1,
        strides=strides,
        use_bias=False,
        padding='same',
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer)
    hidden = hidden + inputs_padded
    return hidden

def resnet_layer(inputs, filters, strides, training, params):
    hidden = resnet_block(inputs, filters, strides, training, params)
    hidden = resnet_block(hidden, filters, 1, training, params)
    hidden = resnet_block(hidden, filters, 1, training, params)
    hidden = resnet_block(hidden, filters, 1, training, params)
    hidden = resnet_block(hidden, filters, 1, training, params)
    hidden = resnet_block(hidden, filters, 1, training, params)
    return hidden

def model_fn(features, labels, mode, params):
    kernel_initializer = tf.initializers.variance_scaling(scale=2.0)
    kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=params['l2_scale'])
    training = (mode == tf.estimator.ModeKeys.TRAIN)
    num_classes = 10

    inputs = features['image'] * 2 - 1

    hidden = tf.layers.conv2d(
        inputs=inputs,
        filters=16,
        kernel_size=3,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer,
        padding='same')

    hidden = resnet_layer(hidden, filters=16, strides=2, training=training, params=params)
    hidden = resnet_layer(hidden, filters=32, strides=1, training=training, params=params)
    hidden = resnet_layer(hidden, filters=64, strides=1, training=training, params=params)

    hidden = batch_norm(
        inputs=hidden,
        momentum=params['bn_momentum'],
        training=training)
    hidden = tf.nn.relu(hidden)

    hidden = tf.reduce_mean(hidden, axis=[1, 2])

    logits = tf.layers.dense(
        inputs=hidden,
        units=num_classes,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer)

    outputs = tf.nn.softmax(logits)
    prediction = tf.argmax(outputs, axis=-1)

    predictions = {
        'prediction': prediction
    }

    loss = None
    if mode != tf.estimator.ModeKeys.PREDICT:
        loss = tf.losses.sparse_softmax_cross_entropy(
            labels=labels['label'],
            logits=logits)
        loss = tf.losses.get_total_loss()

    train_op = None
    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.train.get_or_create_global_step()
        learning_rate = tf.train.piecewise_constant(
            boundaries=params['lr_boundaries'],
            values=params['lr_values'],
            x=global_step)

        optimizer = tf.train.MomentumOptimizer(
            learning_rate=learning_rate,
            momentum=params['momentum'])
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            train_op = optimizer.minimize(
                global_step=global_step,
                loss=loss)

    eval_metric_ops = None
    if mode != tf.estimator.ModeKeys.PREDICT:
        eval_metric_ops = {
            'accuracy': tf.metrics.accuracy(
                predictions=predictions['prediction'],
                labels=labels['label'])
        }

    estimator_spec = tf.estimator.EstimatorSpec(
        loss=loss,
        mode=mode,
        train_op=train_op,
        predictions=predictions,
        eval_metric_ops=eval_metric_ops)
    return estimator_spec
