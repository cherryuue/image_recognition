import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
import pathlib
import efficientnet.keras as efn
from tensorflow.python.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from efficientnet.keras import center_crop_and_resize, preprocess_input
from sklearn.metrics import confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import scipy.io
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import EfficientNetB0

#预处理参数
batch_size = 64
img_height = 224
img_width = 224
ratio = 0.2
#模型运行步数
epochs = 5
num_classes = 102

#处理数据
class_names = ['pink primrose', 'hard-leaved pocket orchid', 'canterbury bells', 'sweet pea', 'english marigold', 'tiger lily', 'moon orchid', 'bird of paradise', 'monkshood', 'globe thistle', 'snapdragon', "colt's foot", 'king protea', 'spear thistle', 'yellow iris', 'globe-flower', 'purple coneflower', 'peruvian lily', 'balloon flower', 'giant white arum lily', 'fire lily', 'pincushion flower', 'fritillary', 'red ginger', 'grape hyacinth', 'corn poppy', 'prince of wales feathers', 'stemless gentian', 'artichoke', 'sweet william', 'carnation', 'garden phlox', 'love in the mist', 'mexican aster', 'alpine sea holly', 'ruby-lipped cattleya', 'cape flower', 'great masterwort', 'siam tulip', 'lenten rose', 'barbeton daisy', 'daffodil', 'sword lily', 'poinsettia', 'bolero deep blue', 'wallflower', 'marigold', 'buttercup', 'oxeye daisy',
          'common dandelion', 'petunia', 'wild pansy', 'primula', 'sunflower', 'pelargonium', 'bishop of llandaff', 'gaura', 'geranium', 'orange dahlia', 'pink-yellow dahlia?', 'cautleya spicata', 'japanese anemone', 'black-eyed susan', 'silverbush', 'californian poppy', 'osteospermum', 'spring crocus', 'bearded iris', 'windflower', 'tree poppy', 'gazania', 'azalea', 'water lily', 'rose', 'thorn apple', 'morning glory', 'passion flower', 'lotus', 'toad lily', 'anthurium', 'frangipani', 'clematis', 'hibiscus', 'columbine', 'desert-rose', 'tree mallow', 'magnolia', 'cyclamen ', 'watercress', 'canna lily', 'hippeastrum ', 'bee balm', 'ball moss', 'foxglove', 'bougainvillea', 'camellia', 'mallow', 'mexican petunia', 'bromelia', 'blanket flower', 'trumpet creeper', 'blackberry lily']
#分17类
# data_dir = 'C:/Users/cherrytree/Desktop/biye_design/data/oxfordflower17/jpg'
#分102类：
data_dir = 'C:/Users/cherrytree/Desktop/biye_design/data/oxford102/oxfordflower102/jpg'

#标签
mat_data = scipy.io.loadmat('imagelabels102.mat')
labels = mat_data['labels'].flatten()
labels = labels -1

# 获取所有图像文件的路径
image_files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.jpg')])

# 加载所有图像
images = []
for img_path in image_files:
    img = image.load_img(img_path, target_size=(img_height, img_width))  # 加载并调整图像大小
    img_array = image.img_to_array(img)  # 转换为数组
    # img_array = img_array / 255.0  # 归一化到[0, 1]区间
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


#图像增强
train_datagen = ImageDataGenerator(
    featurewise_center = True,
    featurewise_std_normalization=True,
    rotation_range = 20,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    horizontal_flip=True)
val_datagen = ImageDataGenerator()

train_generator = train_datagen.flow(X_train,y_train,batch_size = batch_size,shuffle=True)
val_generator = val_datagen.flow(X_test, y_test, batch_size=batch_size, shuffle=True)

print(train_generator)
base_model = EfficientNetB0(weights = 'imagenet',include_top=False,input_shape=[img_height,img_width,3])
base_model.trainable = True

model = Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),  # 加入全局平均池化层
    layers.Dense(num_classes,activation='softmax')
])

model.summary()

# 训练周期的训练和验证准确率，请将 metrics 参数传递给 Model.compile
model.compile(optimizer='adam',
          loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
          metrics=['accuracy'])

history = model.fit(train_generator,validation_data = val_generator ,epochs=epochs)

model.save("models/oxford_102_B0.h5")
model =tf.keras.models.load_model("models/oxford_102_B0.h5")

# #呈现训练结果
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8,8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
