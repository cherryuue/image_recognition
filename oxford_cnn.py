import glob
import os
import math
import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
import pathlib
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import scipy.io
from sklearn.model_selection import train_test_split
import input_data

# Thanks to user m-co for deducing these from the Oxford 102 dataset website.

class_names = ['pink primrose', 'hard-leaved pocket orchid', 'canterbury bells', 'sweet pea', 'english marigold', 'tiger lily', 'moon orchid', 'bird of paradise', 'monkshood', 'globe thistle', 'snapdragon', "colt's foot", 'king protea', 'spear thistle', 'yellow iris', 'globe-flower', 'purple coneflower', 'peruvian lily', 'balloon flower', 'giant white arum lily', 'fire lily', 'pincushion flower', 'fritillary', 'red ginger', 'grape hyacinth', 'corn poppy', 'prince of wales feathers', 'stemless gentian', 'artichoke', 'sweet william', 'carnation', 'garden phlox', 'love in the mist', 'mexican aster', 'alpine sea holly', 'ruby-lipped cattleya', 'cape flower', 'great masterwort', 'siam tulip', 'lenten rose', 'barbeton daisy', 'daffodil', 'sword lily', 'poinsettia', 'bolero deep blue', 'wallflower', 'marigold', 'buttercup', 'oxeye daisy',
          'common dandelion', 'petunia', 'wild pansy', 'primula', 'sunflower', 'pelargonium', 'bishop of llandaff', 'gaura', 'geranium', 'orange dahlia', 'pink-yellow dahlia?', 'cautleya spicata', 'japanese anemone', 'black-eyed susan', 'silverbush', 'californian poppy', 'osteospermum', 'spring crocus', 'bearded iris', 'windflower', 'tree poppy', 'gazania', 'azalea', 'water lily', 'rose', 'thorn apple', 'morning glory', 'passion flower', 'lotus', 'toad lily', 'anthurium', 'frangipani', 'clematis', 'hibiscus', 'columbine', 'desert-rose', 'tree mallow', 'magnolia', 'cyclamen ', 'watercress', 'canna lily', 'hippeastrum ', 'bee balm', 'ball moss', 'foxglove', 'bougainvillea', 'camellia', 'mallow', 'mexican petunia', 'bromelia', 'blanket flower', 'trumpet creeper', 'blackberry lily']
#分17类
# data_dir = 'C:/Users/cherrytree/Desktop/biye_design/data/oxfordflower17/jpg'
#分102类：
data_dir = 'C:/Users/cherrytree/Desktop/biye_design/data/oxford102/oxfordflower102/jpg'
img_height = 100
img_width = 100
ratio = 0.2
num_classes = 102
epochs = 20

#标签
mat_data = scipy.io.loadmat('imagelabels102.mat')
labels = mat_data['labels'].flatten()
labels = labels -1


def grad_cam(model, img, class_idx, conv_layer_name='conv2d_2', target_size=(224, 224)):
    # 获取模型的倒数第二层卷积层
    conv_layer = model.get_layer(conv_layer_name)

    with tf.GradientTape() as tape:
        # 计算图像经过模型的输出
        last_conv_layer_output = conv_layer(img)
        last_conv_layer_output = tf.cast(last_conv_layer_output, tf.float32)

        # 计算目标类别的得分
        model_out = model(img)
        class_channel = model_out[:, class_idx]

    # 计算梯度
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # 计算梯度加权
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # 将加权梯度应用到卷积层输出
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, last_conv_layer_output), axis=-1)

    # 应用ReLU激活函数
    heatmap = np.maximum(heatmap, 0)

    # 归一化热力图
    heatmap /= np.max(heatmap)

    # 调整热力图大小
    heatmap = np.uint8(255 * heatmap)
    heatmap = tf.image.resize(heatmap, target_size)  # 调整为与原图大小一致

    # 叠加热力图到原图
    img = np.array(img)
    superimposed_img = img * 0.6 + heatmap[..., np.newaxis] * 0.4  # 叠加比例

    return superimposed_img, heatmap

# 获取所有图像文件的路径
image_files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.jpg')])

# 加载所有图像
images = []
for img_path in image_files:
    img = image.load_img(img_path, target_size=(img_height, img_width))  # 加载并调整图像大小
    img_array = image.img_to_array(img)  # 转换为数组
    img_array = img_array / 255.0  # 归一化到[0, 1]区间
    images.append(img_array)

# 将图像列表转换为 NumPy 数组
images = np.array(images)
labels = np.array(labels)  # 标签数组

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# 查看划分后的数据集大小
print(f"Training set size: {X_train.shape}")
print(len(y_train))
print(f"Test set size: {X_test.shape}")

# 数据增强部分-数据过拟合处理
data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal",
                          input_shape=(img_height,img_width,3)),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        # layers.RandomContrast(0.1),
    ]
)

model = Sequential([
data_augmentation,
# layers.Rescaling(1./255),
layers.Conv2D(16,3,padding='same', activation='relu'),
layers.MaxPooling2D(),
layers.Conv2D(32, 3, padding='same', activation='relu'),
layers.MaxPooling2D(),
layers.Conv2D(64, 3, padding='same', activation='relu'),
layers.MaxPooling2D(),
# layers.Dropout(0.2),
layers.Flatten(),
layers.Dense(128, activation='relu'),
# layers.Dropout(0.3),
layers.Dense(num_classes, activation='softmax', name="output")
])

#模型优化-编译 adam优化器、SparseCategoricalCrossentropy 损失函数、
# 训练周期的训练和验证准确率，请将 metrics 参数传递给 Model.compile
model.compile(optimizer='adam',
          loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
          metrics=['accuracy'])

#输出模型层数
model.summary()
img = X_test[0]
# 示例：生成并显示热力图
superimposed_img, heatmap = grad_cam(model, img, class_idx=0)
plt.imshow(superimposed_img / 255.0)  # 归一化显示
plt.axis('off')
plt.show()

model.fit(X_train,y_train,epochs=epochs)
test_loss,test_acc = model.evaluate(X_test,y_test,verbose=2)
