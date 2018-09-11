from scipy import misc
import numpy as np
import os 

imagenet_dir = '/home/doi6/Documents/Guangyu/tiny-imagenet-200'

train_dir = os.path.join(imagenet_dir,'train')
test_dir = os.path.join(imagenet_dir,'test')
val_dir = os.path.join(imagenet_dir,'val')
import re

import re

def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)] 
    return sorted(l,key=alphanum_key)


name_to_id = {}
id_to_name = {}
train_imgs = []
train_labels = []
for i, name_i in enumerate(os.listdir(train_dir)):
    id_to_name[i] = name_i
    name_to_id[name_i] = i
    data_dir = os.path.join(train_dir, name_i, 'images')
#     for fn in os.listdir(data_dir):
#         img_i = misc.imread(os.path.join(data_dir,fn), mode='RGB')
#         train_imgs += [np.expand_dims(img_i,0)]
#         train_labels += [i]
# train_imgs = np.concatenate(train_imgs,0)
# train_labels = np.asarray(train_labels)
# train_data = {'image': train_imgs, 'label': train_labels, 'id_to_name': id_to_name, 'name_to_id': name_to_id}
# np.save('train_data', train_data)
val_imgs = []
val_labels = []
data_dir = os.path.join(val_dir,'images')
f = open(os.path.join(val_dir,'val_annotations.txt'))
f=f.read()
f = f.split()
file = natural_sort(os.listdir(data_dir))
for ii,name_ii in enumerate(file):
    label = name_to_id[f[1+6*(ii)]]
    img_i = misc.imread(os.path.join(data_dir,name_ii), mode='RGB')
    val_imgs += [np.expand_dims(img_i,0)]
    val_labels += [label]

val_imgs = np.concatenate(val_imgs,0)
val_labels = np.asarray(val_labels)
val_data = {'image': val_imgs, 'label': val_labels, 'id_to_name': id_to_name, 'name_to_id': name_to_id}
np.save('val_data', val_data)