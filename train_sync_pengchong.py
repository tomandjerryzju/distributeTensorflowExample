import os
import json
import tensorflow as tf
import nets
from tensorflow.keras import layers

tf.app.flags.DEFINE_string("ps_hosts", "", "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("worker_hosts", "", "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("job_name", "", "One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer('task_index', 0, 'task id.')
tf.app.flags.DEFINE_integer('replicas_to_aggregate', 64, 'replicas_to_aggregate.')
tf.app.flags.DEFINE_integer('total_num_replicas', 64, 'total_num_replicas.')
tf.app.flags.DEFINE_float('init_learning_rate', 0.032, 'init_learning_rate.')
tf.app.flags.DEFINE_string("train_dir", "", "train_dir")
tf.app.flags.DEFINE_string("tfrecord_path", "", "tfrecord_path")
tf.app.flags.DEFINE_integer('num_total_image', 58027693, 'num_total_image.')
tf.app.flags.DEFINE_integer('batch_size', 16, 'batch size.')
tf.app.flags.DEFINE_integer('num_epochs', 100, 'num_epochs.')
tf.app.flags.DEFINE_integer('shuffle_buffer_size', 2, 'shuffle_buffer_size.')
tf.app.flags.DEFINE_float('decay_epoch', 1, 'decay_epoch.')
tf.app.flags.DEFINE_float('decay_factor', 0.5, 'decay_factor.')


FLAGS = tf.app.flags.FLAGS


tf.logging.set_verbosity(tf.logging.INFO)
tf.logging.info(tf.__version__)

tfrecord_path = FLAGS.tfrecord_path
eval_tfrecord_path = "your_path"
replicas_to_aggregate=FLAGS.replicas_to_aggregate
total_num_replicas=FLAGS.total_num_replicas
batch_size = FLAGS.batch_size
num_epochs = FLAGS.num_epochs 
num_total_image = FLAGS.num_total_image
steps_per_epoch = int(num_total_image/(batch_size*replicas_to_aggregate))
shuffle_buffer_size = 2
init_learning_rate = FLAGS.init_learning_rate
decay_steps = int(steps_per_epoch*FLAGS.decay_epoch)
decay_factor = FLAGS.decay_factor
save_checkpoints_steps = int(steps_per_epoch/8)
log_step_count_steps = 100
save_summary_steps = 100
keep_checkpoint_max = 50
max_step = num_epochs*steps_per_epoch
train_dir=FLAGS.train_dir

def parse_image(example_proto):
    features = {"image/encoded": tf.FixedLenFeature((), tf.string, default_value=""),
                "image/pickey": tf.FixedLenFeature((), tf.string, default_value=""),
                "image/picurl": tf.FixedLenFeature((), tf.string, default_value=""),
                "label": tf.FixedLenFeature((), tf.int64, default_value=""),}
    parsed_features = tf.parse_single_example(example_proto, features)
    image = parsed_features["image/encoded"]
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.expand_dims(image, 0)
    image = tf.image.resize_bilinear(image, [256, 256],align_corners=False)
    image = tf.squeeze(image, [0])
    label = parsed_features["label"]
    return image,label


def get_dataset(tfrecord_path,batch_size,num_epochs,shuffle_buffer_size):
    tfrecords_dir = tf.gfile.ListDirectory(tfrecord_path)
    filenames = []
    tf.logging.info("has %d tfrecords_dir"%(len(tfrecords_dir)))
    tf.logging.info(tfrecords_dir)
    for dir_name in tfrecords_dir:
        if "tfrecord" in dir_name:
            tfrecord_names = tf.gfile.ListDirectory("%s/%s"%(tfrecord_path,dir_name))
            for tfrecord_name in tfrecord_names:
                if "part" in tfrecord_name:
                    filename = "%s/%s/%s" % (tfrecord_path,dir_name,tfrecord_name)
                    filenames.append(filename)
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.shard(total_num_replicas,FLAGS.task_index)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.shuffle(shuffle_buffer_size)
    dataset = dataset.apply(tf.contrib.data.map_and_batch(parse_image, batch_size))
    dataset = dataset.prefetch(buffer_size=2)
    return dataset



def train_input_fn(tfrecord_path,batch_size,num_epochs,shuffle_buffer_size):
    dataset = get_dataset(tfrecord_path,batch_size,num_epochs,shuffle_buffer_size)
    #iterator = dataset.make_one_shot_iterator()
    #images,labels = iterator.get_next()
    return dataset

def eval_input_fn(tfrecord_path,batch_size):
    tfrecords_dir = tf.gfile.ListDirectory(tfrecord_path)
    filenames = []
    tf.logging.info("has %d tfrecords_dir"%(len(tfrecords_dir)))
    tf.logging.info(tfrecords_dir)
    for dir_name in tfrecords_dir:
        if "tfrecord" in dir_name:
            tfrecord_names = tf.gfile.ListDirectory("%s/%s"%(tfrecord_path,dir_name))
            for tfrecord_name in tfrecord_names:
                if "part" in tfrecord_name:
                    filename = "%s/%s/%s" % (tfrecord_path,dir_name,tfrecord_name)
                    filenames.append(filename)
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.apply(tf.contrib.data.map_and_batch(parse_image, batch_size))
    dataset = dataset.prefetch(buffer_size=2)
    return dataset



def model_fn(features, labels, mode):
    images = features
    predicts = nets.model(images)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predicts)
    loss = tf.losses.mean_squared_error(labels,predicts)
    tf.summary.scalar('loss', loss)
 
    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.train.get_global_step()
        lr = tf.train.exponential_decay(learning_rate=init_learning_rate,global_step=global_step,
                                        decay_steps=decay_steps,decay_rate=decay_factor,staircase=True)
        tf.summary.scalar('lr', lr)
        opt = tf.train.AdamOptimizer(learning_rate=lr)
        train_hooks = []
        optimizer = tf.train.SyncReplicasOptimizer(opt, replicas_to_aggregate=replicas_to_aggregate,total_num_replicas=total_num_replicas)
        train_op = optimizer.minimize(loss=loss,global_step=global_step)
        if FLAGS.job_name == "worker" and FLAGS.task_index==0:
            is_chief=True
        else:
            is_chief=False
        sync_replicas_hook = optimizer.make_session_run_hook(is_chief)
        train_hooks.append(sync_replicas_hook)
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op, training_hooks=train_hooks)
 
    # Add evaluation metrics (for EVAL mode)
    if mode == tf.estimator.ModeKeys.EVAL:
        eval_metric_ops = {"eval_loss": loss}
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)



def train():
    sess_config = tf.ConfigProto(allow_soft_placement=False,log_device_placement=False)
    sess_config.gpu_options.allow_growth = True
    config = tf.estimator.RunConfig(save_checkpoints_steps=save_checkpoints_steps, 
                                    keep_checkpoint_max=keep_checkpoint_max,
                                    save_summary_steps=save_summary_steps,
                                    log_step_count_steps=log_step_count_steps,
                                    session_config=sess_config)
 
    estimator = tf.estimator.Estimator(model_fn=model_fn, model_dir=train_dir,config=config)
    train_spec = tf.estimator.TrainSpec(input_fn=lambda:train_input_fn(tfrecord_path,batch_size,num_epochs,shuffle_buffer_size))
    eval_spec = tf.estimator.EvalSpec(input_fn=lambda:eval_input_fn(eval_tfrecord_path,batch_size))
 
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


def set_environment():
    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")
    tf.logging.info("PS hosts are: {}".format(ps_hosts))
    tf.logging.info("Worker hosts are: {}".format(worker_hosts))
    job_name = FLAGS.job_name
    task_index = FLAGS.task_index
    cluster = {'chief': [worker_hosts[0]], 'worker': worker_hosts[1:], 'ps': ps_hosts}
    if job_name == "worker":
        if task_index==0:
            os.environ['TF_CONFIG'] = json.dumps({'cluster': cluster, 'task': {'type': 'chief', 'index': 0}})
        else:
            os.environ['TF_CONFIG'] = json.dumps({'cluster': cluster, 'task': {'type': 'worker', 'index': task_index-1}})
    elif job_name == "ps":
        os.environ['TF_CONFIG'] = json.dumps({'cluster': cluster, 'task': {'type': 'ps', 'index': task_index}})

    tf.logging.info(cluster)
    



set_environment()
train()