#coding=utf-8
import numpy as np
import tensorflow as tf

# Define parameters
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_float('learning_rate', 0.00003, 'Initial learning rate.')
tf.app.flags.DEFINE_integer('steps_to_validate', 1000,
                     'Steps to validate and print loss')

# For distributed
tf.app.flags.DEFINE_string("ps_hosts", "",
                           "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("worker_hosts", "",
                           "Comma-separated list of hostname:port pairs")
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
  server = tf.train.Server(cluster,job_name=FLAGS.job_name,task_index=FLAGS.task_index)

  issync = FLAGS.issync
  is_chief = (FLAGS.task_index == 0)
  if FLAGS.job_name == "ps":
    server.join()
  elif FLAGS.job_name == "worker":
    with tf.device(tf.train.replica_device_setter(
                    worker_device="/job:worker/task:%d" % FLAGS.task_index,
                    cluster=cluster)):
      global_step = tf.Variable(0, name='global_step', trainable=False)

      input = tf.placeholder("float")
      label = tf.placeholder("float")

      weight = tf.get_variable("weight", [1], tf.float32, initializer=tf.random_normal_initializer())
      biase  = tf.get_variable("biase", [1], tf.float32, initializer=tf.random_normal_initializer())
      pred = tf.multiply(input, weight) + biase

      loss_value = loss(label, pred)
      opt = tf.train.AdamOptimizer(FLAGS.learning_rate)

      if issync == 1:
        #同步模式计算更新梯度
        opt = tf.train.SyncReplicasOptimizer(opt,
                                             replicas_to_aggregate=len(worker_hosts),
                                             total_num_replicas=len(worker_hosts))
      train_op = opt.minimize(loss_value, global_step=global_step)
      if issync == 1:
        if FLAGS.sync_replicas:
          local_init_op = opt.local_step_init_op
          if is_chief:
            local_init_op = opt.chief_init_op

          ready_for_local_init_op = opt.ready_for_local_init_op
          chief_queue_runner = opt.get_chief_queue_runner()
          sync_init_op = opt.get_init_tokens_op()

      init_op = tf.global_variables_initializer()
      saver = tf.train.Saver()
      tf.summary.scalar('cost', loss_value)
      summary_op = tf.summary.merge_all()

      if issync == 1:
        sv = tf.train.Supervisor(
          is_chief=is_chief,
          logdir="./checkpoint/",
          init_op=init_op,
          summary_op=None,
          local_init_op=local_init_op,
          ready_for_local_init_op=ready_for_local_init_op,
          recovery_wait_secs=1,
          saver=saver,
          global_step=global_step,
          save_model_secs=60)
      else:
        sv = tf.train.Supervisor(
          is_chief=is_chief,
          logdir="./checkpoint/",
          init_op=init_op,
          summary_op=None,
          recovery_wait_secs=1,
          saver=saver,
          global_step=global_step,
          save_model_secs=60)

      sess_config = tf.ConfigProto(allow_soft_placement=True,
                                   log_device_placement=False)

      if is_chief:
        print("Worker %d: Initializing session..." % FLAGS.task_index)
      else:
        print("Worker %d: Waiting for session to be initialized..." %
              FLAGS.task_index)

    with sv.prepare_or_wait_for_session(server.target, config=sess_config) as sess:
      print("Worker %d: Session initialization complete." % FLAGS.task_index)
      # 如果是同步模式
      if is_chief and issync == 1:
        sess.run(sync_init_op)
        sv.start_queue_runners(sess, [chief_queue_runner])

      step = 0
      while  step < 100000000:
        train_x = np.random.randn(1)
        train_y = 2 * train_x + np.random.randn(1) * 0.33  + 10
        _, loss_v, step = sess.run([train_op, loss_value,global_step], feed_dict={input:train_x, label:train_y})
        if step % steps_to_validate == 0:
          w,b = sess.run([weight,biase])
          print("step: %d, weight: %f, biase: %f, loss: %f" %(step, w, b, loss_v))

    sv.stop()

def loss(label, pred):
  return tf.square(label - pred)


if __name__ == "__main__":
  tf.app.run()

















