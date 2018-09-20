import tensorflow as tf
from Xception import *
import numpy as np
import os
import tflearn
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

LAMBDA = 0
CENTER_LOSS_ALPHA = 0.5
NUM_CLASSES = 200
batch_size = 128
    
global_step = tf.Variable(0, trainable=False, name='global_step')

with tf.name_scope('input'):
    input_images = tf.placeholder(tf.float32,  shape = (batch_size, 64, 64, 3), name = 'input_images')
    labels = tf.placeholder(tf.int64, shape=(batch_size), name='labels')

def get_center_loss(features, labels, alpha, num_classes):
    """获取center loss及center的更新op
    
    Arguments:
        features: Tensor,表征样本特征,一般使用某个fc层的输出,shape应该为[batch_size, feature_length].
        labels: Tensor,表征样本label,非one-hot编码,shape应为[batch_size].
        alpha: 0-1之间的数字,控制样本类别中心的学习率,细节参考原文.
        num_classes: 整数,表明总共有多少个类别,网络分类输出有多少个神经元这里就取多少.
    
    Return：
        loss: Tensor,可与softmax loss相加作为总的loss进行优化.
        centers: Tensor,存储样本中心值的Tensor，仅查看样本中心存储的具体数值时有用.
        centers_update_op: op,用于更新样本中心的op，在训练时需要同时运行该op，否则样本中心不会更新
    """
    # 获取特征的维数，例如256维
    len_features = features.get_shape()[1]
    # 建立一个Variable,shape为[num_classes, len_features]，用于存储整个网络的样本中心，
    # 设置trainable=False是因为样本中心不是由梯度进行更新的
    centers = tf.get_variable('centers', [num_classes, len_features], dtype=tf.float32,
        initializer=tf.constant_initializer(0), trainable=False)
    # 将label展开为一维的，输入如果已经是一维的，则该动作其实无必要
    labels = tf.reshape(labels, [-1])
    
    # 根据样本label,获取mini-batch中每一个样本对应的中心值
    centers_batch = tf.gather(centers, labels)
    
    # debug
    features =  tf.reshape(features,[128,-1])
    features = features[:,:2]
    print(features.get_shape())
    
    # 计算loss
    loss = tf.nn.l2_loss(features - centers_batch)
    
    # 当前mini-batch的特征值与它们对应的中心值之间的差
    diff = centers_batch - features
    
    # 获取mini-batch中同一类别样本出现的次数,了解原理请参考原文公式(4)
    unique_label, unique_idx, unique_count = tf.unique_with_counts(labels)
    appear_times = tf.gather(unique_count, unique_idx)
    appear_times = tf.reshape(appear_times, [-1, 1])
    
    diff = diff / tf.cast((1 + appear_times), tf.float32)
    diff = alpha * diff
    
    centers_update_op = tf.scatter_sub(centers, labels, diff)
    
    return loss, centers, centers_update_op


def build_network(input_images, labels, ratio=0, reuse=False):
    logits, feature_list = XceptionModel(input_images, 200, is_training = True, data_format='channels_last')
    
    with tf.name_scope('loss'):
        with tf.variable_scope('center_loss1'):
            center_loss1, centers1, centers_update_op1 = get_center_loss(feature_list[0], labels, CENTER_LOSS_ALPHA, NUM_CLASSES)
        with tf.variable_scope('center_loss2'):
            center_loss2, centers2, centers_update_op2 = get_center_loss(feature_list[1], labels, CENTER_LOSS_ALPHA, NUM_CLASSES)
        with tf.name_scope('softmax_loss'):
            softmax_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))
        with tf.name_scope('total_loss'):
            total_loss = softmax_loss# + ratio * center_loss1#(center_loss1*0.8 + center_loss2*0.2) * 4
    
    with tf.name_scope('acc'):
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(logits, 1), labels), tf.float32))
    
    with tf.name_scope('loss/'):
        tf.summary.scalar('CenterLoss1', center_loss1)
        tf.summary.scalar('CenterLoss2', center_loss2)
        tf.summary.scalar('SoftmaxLoss', softmax_loss)
        tf.summary.scalar('TotalLoss', total_loss)
        
    centers_update_op_list = [centers_update_op1, centers_update_op2]
    return logits, feature_list, total_loss, accuracy, centers_update_op_list




logits, feature_list, total_loss, accuracy, centers_update_op_list = build_network(input_images, labels, ratio=LAMBDA)
features = feature_list[0]
centers_update_op1 = centers_update_op_list[0]
centers_update_op2 = centers_update_op_list[1]

train_images =  np.load('/home/doi6/Documents/Guangyu/Xception/train_images.npy',encoding=('latin1'))
train_labels =  np.load('/home/doi6/Documents/Guangyu/Xception/train_labels.npy',encoding=('latin1'))

from random import shuffle
idx = list(range(len(train_images)))
shuffle(idx)
train_images = train_images[idx] 
train_labels = train_labels[idx]

train_images = train_images[10000:]
train_labels = train_labels[10000:]
val_images = train_images[:10000] 
val_labels = train_labels[:10000]

val_images = np.load('/home/doi6/Documents/Guangyu/tiny-imagenet-200/val_data.npy',encoding=('latin1')).item()['image']
val_labels = np.load('/home/doi6/Documents/Guangyu/tiny-imagenet-200/val_data.npy',encoding=('latin1')).item()['label'] 


idx_v = list(range(len(val_images)))
shuffle(idx_v)
test_images = val_images[idx_v] 
test_labels = val_labels[idx_v]

not_restore = ['fully_connected/biases:0','fully_connected/weights:0','fully_connected_1/biases:0','fully_connected_1/weights:0',
'center_loss1/centers:0','center_loss2/centers:0','global_step:0','dense/kernel:0','dense/bias:0']

restore_var = [v for v in tf.trainable_variables() if v.name not in not_restore]
#print(restore_var)

var_list = [v for v in tf.trainable_variables() if v.name not in restore_var]

graph = tf.get_default_graph()

#graph = tf.get_default_graph()
#graph = graph.as_default()
#graph.__enter__()

optimizer = tf.train.AdamOptimizer(0.0001)
with tf.control_dependencies([centers_update_op1, centers_update_op2]):
    train_op = optimizer.minimize(total_loss, global_step=global_step)#var_list=var_list)
summary_op = tf.summary.merge_all()
sess = tf.Session(graph=graph)
saver = tf.train.Saver(restore_var)
init = tf.global_variables_initializer()
sess.run(init)
saver.restore(sess, "./checkpoints/xception_model.ckpt")

for ep_i in range(100):
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


saver.save(sess,'./model_save/center_loss.ckpt')
































