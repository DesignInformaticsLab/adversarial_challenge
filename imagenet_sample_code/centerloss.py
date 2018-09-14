LAMBDA = 1e-1
CENTER_LOSS_ALPHA = 0.5
NUM_CLASSES = 200
batch_size = 128


import os
import numpy as np
import tensorflow as tf
import tflearn
slim = tf.contrib.slim
os.environ['CUDA_VISIBLE_DEVICES'] = '0'



with tf.name_scope('input'):
    input_images = tf.placeholder(tf.float32, shape=(batch_size,64,64,3), name='input_images')
    labels = tf.placeholder(tf.int64, shape=(batch_size), name='labels')
    
global_step = tf.Variable(0, trainable=False, name='global_step')

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

def inference(input_images, num_class=200, reuse=False):
    with slim.arg_scope([slim.conv2d], kernel_size=3, padding='SAME'):
        with slim.arg_scope([slim.max_pool2d], kernel_size=2):

            x = slim.conv2d(input_images, num_outputs=64, scope='conv1_1')
            x = slim.conv2d(x, num_outputs=64, scope='conv1_2')
            x = slim.max_pool2d(x, scope='pool1')

            x = slim.conv2d(x, num_outputs=128, scope='conv2_1')
            x = slim.conv2d(x, num_outputs=128, scope='conv2_2')
            x = slim.max_pool2d(x, scope='pool2')

            x = slim.conv2d(x, num_outputs=256, scope='conv3_1')
            x = slim.conv2d(x, num_outputs=256, scope='conv3_2')
            x = slim.max_pool2d(x, scope='pool3')

            x = slim.conv2d(x, num_outputs=512, scope='conv4_1')
            x = slim.conv2d(x, num_outputs=512, scope='conv4_2')
            x = slim.max_pool2d(x, scope='pool3')

            x = slim.flatten(x, scope='flatten')

            feature3 = x = slim.fully_connected(x, num_outputs=512, activation_fn=None, scope='fc0')

            feature2 = x = slim.fully_connected(x, num_outputs=32, activation_fn=None, scope='fc1')

            feature1 = x =slim.fully_connected(x, num_outputs=2, activation_fn=None, scope='fc2')

            x = tflearn.prelu(feature3)

            x = slim.fully_connected(x, num_outputs=num_class, activation_fn=None, scope='fc3')

    feature_list = [feature1, feature2]
    return x, feature_list


def build_network(input_images, labels, ratio=0.5, reuse=False):
    logits, feature_list = inference(input_images, num_class=NUM_CLASSES)
    
    with tf.name_scope('loss'):
        with tf.variable_scope('center_loss1'):
            center_loss1, centers1, centers_update_op1 = get_center_loss(feature_list[0], labels, CENTER_LOSS_ALPHA, NUM_CLASSES)
        with tf.variable_scope('center_loss2'):
            center_loss2, centers2, centers_update_op2 = get_center_loss(feature_list[1], labels, CENTER_LOSS_ALPHA, NUM_CLASSES)
        with tf.name_scope('softmax_loss'):
            softmax_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))
        with tf.name_scope('total_loss'):
            total_loss = softmax_loss + ratio * center_loss1#(center_loss1*0.8 + center_loss2*0.2) * 4
    
    with tf.name_scope('acc'):
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(logits, 1), labels), tf.float32))
    
    with tf.name_scope('loss/'):
        tf.summary.scalar('CenterLoss1', center_loss1)
        tf.summary.scalar('CenterLoss2', center_loss2)
        tf.summary.scalar('SoftmaxLoss', softmax_loss)
        tf.summary.scalar('TotalLoss', total_loss)
        
    centers_update_op_list = [centers_update_op1, centers_update_op2]
    return logits, feature_list, total_loss, accuracy, centers_update_op_list



with tf.variable_scope("build_network", reuse=False):
    logits, feature_list, total_loss, accuracy, centers_update_op_list = build_network(input_images, labels, ratio=LAMBDA)
features = feature_list[0]
centers_update_op1 = centers_update_op_list[0]
centers_update_op2 = centers_update_op_list[1]



train_images = np.load('/home/doi6/Documents/Guangyu/tiny-imagenet-200/train_data.npy',encoding=('latin1')).item()['image'] / 255.
train_labels = np.load('/home/doi6/Documents/Guangyu/tiny-imagenet-200/train_data.npy',encoding=('latin1')).item()['label'] 

if 0:
    train_image_5 = train_images[train_labels==5]
    train_image_7 = train_images[train_labels==7]
    train_images = np.concatenate([train_image_5, train_image_7],0)
    train_labels = np.asarray( [0]*len(train_image_5) + [1]*len(train_image_7) )
from random import shuffle
idx = list(range(len(train_images)))
shuffle(idx)
train_images = train_images[idx] 
train_labels = train_labels[idx]

test_images = train_images[:200]
test_labels = train_labels[:200]

val_images = np.load('/home/doi6/Documents/Guangyu/tiny-imagenet-200/val_data.npy',encoding=('latin1')).item()['image'] / 255.
val_labels = np.load('/home/doi6/Documents/Guangyu/tiny-imagenet-200/val_data.npy',encoding=('latin1')).item()['label'] 

if 0:
    val_image_5 = val_images[val_labels==5]
    val_image_7 = val_images[val_labels==7]
    val_images = np.concatenate([val_image_5, val_image_7],0)
    val_labels = np.asarray( [0]*len(val_image_5) + [1]*len(val_image_7) )
idx_v = list(range(len(val_images)))
shuffle(idx_v)
val_images = val_images[idx_v] 
val_labels = val_labels[idx_v]
val_images = val_images[:200]
val_labels = val_labels[:200]

optimizer = tf.train.AdamOptimizer(0.0001)
with tf.control_dependencies([centers_update_op1, centers_update_op2]):
    train_op = optimizer.minimize(total_loss, global_step=global_step)
summary_op = tf.summary.merge_all()
saver = tf.train.Saver()
sess = tf.Session()
sess.run(tf.global_variables_initializer())

step = sess.run(global_step)
for ep_i in range(50):
    train_acc = []
    train_loss = []
    for jj in range(train_images.shape[0]//batch_size):
        _, summary_str, train_acc_i, train_loss_i = sess.run(
            [train_op, summary_op, accuracy, total_loss],
            feed_dict={
                input_images: train_images[jj*batch_size:(1+jj)*batch_size],
                labels: train_labels[jj*batch_size:(1+jj)*batch_size]
            })
        train_acc += [train_acc_i]
        train_loss += [train_loss_i]
    testing_acc = sess.run(accuracy,feed_dict={input_images: val_images[0:128],labels: val_labels[0:128]})
    print(("epoch: {}, train_acc:{:.4f}, train_loss:{:.4f},testing_acc:{:.4f}".
          format(ep_i, np.mean(train_acc), np.mean(train_loss),testing_acc)))


saver.save(sess,'./model_save/center_loss.ckpt')



