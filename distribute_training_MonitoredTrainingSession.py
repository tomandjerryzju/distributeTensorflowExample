# coding=utf-8
import numpy as np
import tensorflow as tf
import time

# Define parameters
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_float('learning_rate', 0.00003, 'Initial learning rate.')
tf.app.flags.DEFINE_integer('steps_to_validate', 100, 'Steps to validate and print loss')

# For distributed
tf.app.flags.DEFINE_string("ps_hosts", "0.0.0.0:2222", "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("worker_hosts", "0.0.0.0:2224,0.0.0.0:2225", "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("job_name", "", "One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
tf.app.flags.DEFINE_integer("issync", 1, "是否采用分布式的同步模式，1表示同步模式，0表示异步模式")

# Hyperparameters
learning_rate = FLAGS.learning_rate
steps_to_validate = FLAGS.steps_to_validate


def main(_):
    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
    server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)

    issync = FLAGS.issync
    is_chief = (FLAGS.task_index == 0)
    if FLAGS.job_name == "ps":
        server.join()
    elif FLAGS.job_name == "worker":
        """
        分布式的启动顺序为:ps -> chief worker -> non chief worker。tf中没有提供控制启动顺序的方法，因此必须在代码中
        强制保证这个启动的顺序，否则会报"OS Error"错误。
        """
        time.sleep(5)
        if not is_chief:
            time.sleep(5)
        with tf.device(tf.train.replica_device_setter(worker_device="/job:worker/task:%d" % FLAGS.task_index, cluster=cluster)):
            global_step = tf.Variable(0, name='global_step', trainable=False)

            input = tf.placeholder("float")
            label = tf.placeholder("float")

            weight = tf.get_variable("weight", [1], tf.float32, initializer=tf.random_normal_initializer())
            biase = tf.get_variable("biase", [1], tf.float32, initializer=tf.random_normal_initializer())
            pred = tf.multiply(input, weight) + biase

            loss_value = loss(label, pred)

            # The StopAtStepHook handles stopping after running given steps.
            hooks = [tf.train.StopAtStepHook(last_step=50000)]

            opt = tf.train.AdamOptimizer(FLAGS.learning_rate)

            if issync == 1:
                # 同步模式计算更新梯度
                opt = tf.train.SyncReplicasOptimizer(opt,
                                                     replicas_to_aggregate=len(worker_hosts),
                                                     total_num_replicas=len(worker_hosts))
                hooks.append(opt.make_session_run_hook((FLAGS.task_index == 0)))

            train_op = opt.minimize(loss_value, global_step=global_step)

            sess_config = tf.ConfigProto(allow_soft_placement=True,
                                         log_device_placement=False)

            # The MonitoredTrainingSession takes care of session initialization,
            # restoring from a checkpoint, saving to a checkpoint, and closing when done
            # or an error occurs.
            with tf.train.MonitoredTrainingSession(master=server.target,
                                                   is_chief=(FLAGS.task_index == 0),
                                                   checkpoint_dir="./checkpoint_dir",
                                                   hooks=hooks,
                                                   config=sess_config) as sess:
                while not sess.should_stop():
                    train_x = np.random.randn(1)
                    train_y = 2 * train_x + np.random.randn(1) * 0.33 + 10
                    _, loss_v, step = sess.run([train_op, loss_value, global_step],
                                               feed_dict={input: train_x, label: train_y})
                    if step % steps_to_validate == 0:
                        w, b = sess.run([weight, biase])
                        print("step: %d, weight: %f, biase: %f, loss: %f" % (step, w, b, loss_v))


def loss(label, pred):
    return tf.square(label - pred)


if __name__ == "__main__":
    """
    python distribute_training_MonitoredTrainingSession.py --job_name=ps --task_index=0 --issync=1
    python distribute_training_MonitoredTrainingSession.py --job_name=worker --task_index=0 --issync=1
    python distribute_training_MonitoredTrainingSession.py --job_name=worker --task_index=1 --issync=1
    """
    tf.app.run()
