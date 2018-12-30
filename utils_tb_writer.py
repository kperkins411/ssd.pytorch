import tensorflow as tf
from datetime import datetime


class Writer:
    '''
    just a class to make it easy to track tensorboard events
    '''
    def __init__(self, log_dir):
        now = datetime.now()
        logdir = log_dir +"/" + now.strftime("%Y%m%d-%H%M%S") + "/"
        self.writer = tf.summary.FileWriter(logdir)
        self.last_batch_iteration = 0

    def __call__(self,name, lr, iter=None):
        '''
        used to write to tensorboard
        :param name: string name of var to track
        :param lr:
        :return:
        '''
        """Log a scalar variable."""
        iter1 = iter if iter is None else self.last_batch_iteration
        summary = tf.Summary(value=[tf.Summary.Value(tag=name, simple_value=lr)])
        self.writer.add_summary(summary, self.last_batch_iteration)
        # print(f"Writer:{self.last_batch_iteration}: lr:{lr} ")
        self.last_batch_iteration+=1