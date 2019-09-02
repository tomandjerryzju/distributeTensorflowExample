# coding=utf-8
"""
参考
官方文档1：https://www.tensorflow.org/guide/estimators?hl=zh-CN  (单机)
官方文档2：https://www.tensorflow.org/guide/premade_estimators?hl=zh-CN  (单机)
示例代码：train_sync_pengchong.py    (分布式)
官方文档3：tf.estimator.train_and_evaluate的API说明    (分布式)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import pandas as pd
import json
import os


# Define parameters
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('batch_size', 100, 'batch size.')
tf.app.flags.DEFINE_string("ps_hosts", "0.0.0.0:2222", "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("worker_hosts", "0.0.0.0:2224,0.0.0.0:2225,0.0.0.0:2226", "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("model_dir", '/Users/hyc/workspace/distributeTensorflowExample/gitignore/model/iris', "model dir")
tf.app.flags.DEFINE_string("job_name", "", "One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
tf.app.flags.DEFINE_integer("issync", 1, "是否采用分布式的同步模式，1表示同步模式，0表示异步模式")

CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth',
                    'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']


def set_environment():
    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")
    tf.logging.info("PS hosts are: {}".format(ps_hosts))
    tf.logging.info("Worker hosts are: {}".format(worker_hosts))
    job_name = FLAGS.job_name
    task_index = FLAGS.task_index
    cluster = {'chief': [worker_hosts[0]], 'worker': worker_hosts[1:-1], 'ps': ps_hosts}    # evaluator不需要在cluster中指定
    if job_name == "worker":
        if task_index == 0:
            os.environ['TF_CONFIG'] = json.dumps({'cluster': cluster, 'task': {'type': 'chief', 'index': 0}})
        elif task_index > 0 and task_index < len(worker_hosts)-1:
            os.environ['TF_CONFIG'] = json.dumps({'cluster': cluster, 'task': {'type': 'worker', 'index': task_index - 1}})  # worker的task_index从0开始，与使用Supervisor和MonitoredTrainingSession实现的分布式的worker配置不同
        else:
            os.environ['TF_CONFIG'] = json.dumps({"cluster": cluster, "task": {"type": "evaluator", "index": 0}})
    elif job_name == "ps":
        os.environ['TF_CONFIG'] = json.dumps({'cluster': cluster, 'task': {'type': 'ps', 'index': task_index}})

    tf.logging.info(cluster)

    return worker_hosts


worker_hosts = set_environment()


def load_data(y_name='Species'):
    train_path = "/Users/hyc/workspace/distributeTensorflowExample/gitignore/dataset/iris_test.csv"
    test_path = "/Users/hyc/workspace/distributeTensorflowExample/gitignore/dataset/iris_training.csv"

    train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
    train_x, train_y = train, train.pop(y_name)
    test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)
    test_x, test_y = test, test.pop(y_name)

    return (train_x, train_y), (test_x, test_y)


def train_input_fn(features, labels, batch_size):
    """An input function for training"""
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    dataset = dataset.shard(len(worker_hosts)-1, FLAGS.task_index)
    dataset = dataset.repeat(10000)
    dataset = dataset.shuffle(1000)
    dataset = dataset.batch(batch_size)
    return dataset


def eval_input_fn(features, labels, batch_size):
    """An input function for evaluation or prediction"""
    features=dict(features)
    if labels is None:
        inputs = features
    else:
        inputs = (features, labels)
    dataset = tf.data.Dataset.from_tensor_slices(inputs)
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)
    return dataset


def model_fn(features, labels, mode, params):
    # defind model
    net = tf.feature_column.input_layer(features, params['feature_columns'])
    net = tf.layers.dense(net, 10, activation=tf.nn.relu)
    net = tf.layers.dense(net, 10, activation=tf.nn.relu)
    logits = tf.layers.dense(net, 3, activation=None)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=tf.nn.softmax(logits))

    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    accuracy = tf.metrics.accuracy(labels=labels, predictions=tf.argmax(logits, 1), name='acc_op')
    metrics = {'accuracy': accuracy}
    tf.summary.scalar('accuracy', accuracy[1])
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.train.get_global_step()
        opt = tf.train.AdagradOptimizer(learning_rate=0.1)
        train_hooks = []
        if FLAGS.job_name == "worker" and FLAGS.task_index==0:
            is_chief=True
        else:
            is_chief=False
        if FLAGS.issync == 1:
            print("同步模式")
            opt = tf.train.SyncReplicasOptimizer(opt, replicas_to_aggregate=len(worker_hosts)-1, total_num_replicas=len(worker_hosts)-1)
            train_hooks.append(opt.make_session_run_hook(is_chief))
        else:
            print("异步模式")
        train_op = opt.minimize(loss=loss, global_step=global_step)

        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op, training_hooks=train_hooks)


def main(argv):
    (train_x, train_y), (test_x, test_y) = load_data()
    my_feature_columns = []
    for key in train_x.keys():
        my_feature_columns.append(tf.feature_column.numeric_column(key=key))

    sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    sess_config.gpu_options.allow_growth = True
    config = tf.estimator.RunConfig(keep_checkpoint_max=5,
                                    log_step_count_steps=1,
                                    session_config=sess_config)
    estimator = tf.estimator.Estimator(model_fn=model_fn,
                                        model_dir=FLAGS.model_dir,
                                        params={'feature_columns': my_feature_columns},
                                        config=config)

    """
    estimator.train/estimator.evaluate/estimator.predict用于单机。同时，单机情况下，需要置issync为1，且不需要运行set_environment。
    """
    # estimator.train(input_fn=lambda: train_input_fn(train_x, train_y, FLAGS.batch_size))
    # eval_result = estimator.evaluate(input_fn=lambda: eval_input_fn(test_x, test_y, FLAGS.batch_size))
    # predict_x = {'SepalLength': [5.1, 5.9, 6.9],
    #              'SepalWidth': [3.3, 3.0, 3.1],
    #              'PetalLength': [1.7, 4.2, 5.4],
    #              'PetalWidth': [0.5, 1.5, 2.1]}
    # predictions = estimator.predict(input_fn=lambda: eval_input_fn(predict_x, labels=None, batch_size=FLAGS.batch_size))

    """
    以下既可用于分布式，也可用于单机，唯一区别在于是否运行set_environment，见tf.estimator.train_and_evaluate的API说明。
    """
    train_spec = tf.estimator.TrainSpec(input_fn=lambda: train_input_fn(train_x, train_y, FLAGS.batch_size))
    eval_spec = tf.estimator.EvalSpec(input_fn=lambda: eval_input_fn(test_x, test_y, FLAGS.batch_size))
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)   # 必须指定eval_spec




if __name__ == '__main__':
    """
    测试成功
    python distribute_estimator.py --job_name=ps --task_index=0 --issync=1
    python distribute_estimator.py --job_name=worker --task_index=0 --issync=1
    python distribute_estimator.py --job_name=worker --task_index=1 --issync=1
    """
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)