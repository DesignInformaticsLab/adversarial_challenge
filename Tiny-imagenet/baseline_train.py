import tensorflow as tf
#from xception_batch import *
from Xception import *
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

NUM_CLASSES = 200
batch_size = 24

with tf.name_scope('input'):
    input_images = tf.placeholder(tf.float32,  shape = (batch_size, 64, 64, 3), name = 'input_images')
    input_images_re = tf.image.resize_nearest_neighbor(input_images,(299,299))
    labels = tf.placeholder(tf.int64, shape=(batch_size), name='labels')


def build_network(input_images, labels, reuse=False):
    logits = XceptionModel(input_images_re, 200, is_training = True, data_format='channels_last')
    
    with tf.name_scope('loss'):
        with tf.name_scope('softmax_loss'):
            total_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))   
    with tf.name_scope('acc'):
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(logits, 1), labels), tf.float32))  
    with tf.name_scope('loss/'):
        tf.summary.scalar('TotalLoss', total_loss)
    return logits, total_loss, accuracy


logits, total_loss, accuracy = build_network(input_images, labels)


train_images =  np.load('/home/doi6/Documents/Guangyu/tiny-imagenet-200/train_data.npy',encoding=('latin1')).item()['image']
train_labels =  np.load('/home/doi6/Documents/Guangyu/tiny-imagenet-200/train_data.npy',encoding=('latin1')).item()['label']


from random import shuffle
idx = list(range(len(train_images)))
shuffle(idx)
train_images = train_images[idx] 
train_labels = train_labels[idx]
train_images = train_images[10000:]
train_labels = train_labels[10000:]
val_images = train_images[:10000] 
val_labels = train_labels[:10000]
test_images = np.load('/home/doi6/Documents/Guangyu/tiny-imagenet-200/val_data.npy',encoding=('latin1')).item()['image']
test_labels = np.load('/home/doi6/Documents/Guangyu/tiny-imagenet-200/val_data.npy',encoding=('latin1')).item()['label'] 


idx_v = list(range(len(test_images)))
shuffle(idx_v)
test_images = test_images[idx_v] 
test_labels = test_labels[idx_v]

#not_restore = ['fully_connected/biases:0','fully_connected/weights:0','fully_connected_1/biases:0','fully_connected_1/weights:0','dense/kernel:0','dense/bias:0']

not_restore = ['fully_connected/biases:0','fully_connected/weights:0','fully_connected_1/biases:0','fully_connected_1/weights:0','dense/kernel:0','dense/bias:0']

restore_var = [v for v in tf.global_variables() if v.name not in not_restore]
#print(restore_var)

#var_list = [v for v in tf.trainable_variables() if v.name not in restore_var]

optimizer = tf.train.AdamOptimizer(0.0001)

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train_op = optimizer.minimize(total_loss)



summary_op = tf.summary.merge_all()
sess = tf.Session()
saver = tf.train.Saver(var_list = tf.global_variables())
restorer = tf.train.Saver(restore_var)
init = tf.global_variables_initializer()
sess.run(init)

#print([op.name for op in tf.get_default_graph().get_operations() if op.op_def and op.op_def.name=='VariableV2'])

restorer.restore(sess, "./checkpoints/xception_model.ckpt")


for ep_i in range(5):
    train_acc = []
    train_loss = []
    testing_acc = []
    val_acc = []
    for jj in range(train_images.shape[0]//batch_size):
        _, summary_str, train_acc_i, train_loss_i = sess.run(
            [train_op, summary_op, accuracy, total_loss],
            feed_dict={
                input_images: train_images[jj*batch_size:(1+jj)*batch_size],
                labels: train_labels[jj*batch_size:(1+jj)*batch_size]
            })


        train_acc += [train_acc_i]
        train_loss += [train_loss_i]
    for jjj in range(val_images.shape[0]//batch_size):
        val_acc_i = sess.run(accuracy,feed_dict={input_images: val_images[jjj*batch_size:(1+jjj)*batch_size],labels: val_labels[jjj*batch_size:(1+jjj)*batch_size]})
        val_acc +=[val_acc_i]
    for jjj in range(test_images.shape[0]//batch_size):
        testing_acc_i = sess.run(accuracy,feed_dict={input_images: test_images[jjj*batch_size:(1+jjj)*batch_size],labels: test_labels[jjj*batch_size:(1+jjj)*batch_size]})
        testing_acc +=[testing_acc_i]
    print(("epoch: {}, train_acc:{:.4f}, train_loss:{:.4f},val_acc:{:.4f},testing_acc:{:.4f}".
          format(ep_i, np.mean(train_acc), np.mean(train_loss),np.mean(val_acc),np.mean(testing_acc))))


saver.save(sess,'./model_save_softmax_fixed/center_loss.ckpt')
































