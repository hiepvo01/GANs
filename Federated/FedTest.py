import nest_asyncio
nest_asyncio.apply()

import collections
import time

import tensorflow as tf

import tensorflow_federated as tff

source, _ = tff.simulation.datasets.emnist.load_data()


def map_fn(example):
  return collections.OrderedDict(
      x=tf.reshape(example['pixels'], [-1, 784]), y=example['label'])


def client_data(n):
  ds = source.create_tf_dataset_for_client(source.client_ids[n])
  return ds.repeat(10).shuffle(500).batch(20).map(map_fn)


train_data = [client_data(n) for n in range(10)]
element_spec = train_data[0].element_spec


def model_fn():
  model = tf.keras.models.Sequential([
      tf.keras.layers.InputLayer(input_shape=(784,)),
      tf.keras.layers.Dense(units=10, kernel_initializer='zeros'),
      tf.keras.layers.Softmax(),
  ])
  return tff.learning.from_keras_model(
      model,
      input_spec=element_spec,
      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
      metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])


trainer = tff.learning.build_federated_averaging_process(
    model_fn, client_optimizer_fn=lambda: tf.keras.optimizers.SGD(0.02))


def evaluate(num_rounds=10):
  state = trainer.initialize()
  for _ in range(num_rounds):
    t1 = time.time()
    state, metrics = trainer.next(state, train_data)
    t2 = time.time()
    print('metrics {m}, round time {t:.2f} seconds'.format(
        m=metrics, t=t2 - t1))

evaluate()