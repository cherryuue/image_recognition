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
from sklearn.metrics import confusion_matrix
import seaborn as sns


#kaggle数据集-5类-3690
data_dir = 'C:/Users/cherrytree/Desktop/biye_design/data/flower_photos'
#模型存储位置：
save_path = 'C:/Users/cherrytree/Desktop/biye_design/machine_learning/my_flower/cnn_save_5'
#预处理参数
batch_size = 32
img_height = 128
img_width = 128
ratio = 0.2

#处理数据
def get_data(img_height, img_width ,data_dir,ratio):
    data_dir = pathlib.Path(data_dir).with_suffix('')
    # train
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=ratio,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    # valid
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=ratio,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),  # 调整大小，还可以使用layers.Resizing
        batch_size=batch_size)

    class_names = train_ds.class_names
    print(class_names)
    num_classes = len(class_names)
    return train_ds,val_ds,num_classes

train_ds ,val_ds ,num_classes = get_data(img_height, img_width ,data_dir,ratio)

#打乱数据集，配置数据集，保存在内存中 data.chche()
AUTOTUNE = tf.data.AUTOTUNE#？？？
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size = AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size = AUTOTUNE)

# 数据增强部分-数据过拟合处理
data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal",
                          input_shape=(img_height,img_width,3)),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
    ]
)

model = Sequential([
data_augmentation,
layers.Rescaling(1./255),
layers.Conv2D(16,3,padding='same', activation='relu'),
layers.MaxPooling2D(),
layers.Conv2D(32, 3, padding='same', activation='relu'),
layers.MaxPooling2D(),
layers.Conv2D(64, 3, padding='same', activation='relu'),
layers.MaxPooling2D(),
layers.Dropout(0.2),
layers.Flatten(),
layers.Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
layers.Dropout(0.3),
layers.Dense(num_classes, activation='softmax', name="output")
])

#模型优化-编译 adam优化器、SparseCategoricalCrossentropy 损失函数、
# 训练周期的训练和验证准确率，请将 metrics 参数传递给 Model.compile
model.compile(optimizer='adam',
          loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
          metrics=['accuracy'])

#输出模型层数
model.summary()

# #训练模型,训练10个周期
epochs = 20#25
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)
#模型保存
model.save("models/cnn_save.h5")

model = tf.keras.models.load_model("models/cnn_save.h5")
#
# 生成混淆矩阵
true_labels = []
predicted_labels = []

for images, labels in val_ds:
    predictions = model(images, training=False)
    predicted_class = np.argmax(predictions, axis=1)
    true_labels.extend(labels.numpy())
    predicted_labels.extend(predicted_class)

true_labels = np.array(true_labels)
predicted_labels = np.array(predicted_labels)

# 生成混淆矩阵
cm = confusion_matrix(true_labels, predicted_labels)

# 计算每行的总和
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# 将混淆矩阵值转换为百分比
cm_percentage = cm_normalized

# 输出百分比形式的混淆矩阵
print("Confusion Matrix :")
print(cm_percentage)

# 可视化百分比形式的混淆矩阵
plt.figure(figsize=(10, 5))
sns.heatmap(cm_percentage, annot=True, fmt='.2f', cmap='Purples', xticklabels=val_ds.class_names, yticklabels=val_ds.class_names)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix (Percentages)')
plt.xticks(rotation=90)  # 旋转X轴标签
plt.show()

#呈现训练结果
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

# #标准化数据，将RGB通道的255归一化 data.map()
# normalization_layer = layers.Rescaling(1./255)
# normalization_ds = train_ds.map(lambda x,y:(normalization_layer(x),y))
# image_batch,labels_batch = (next(iter(normalization_ds)))
# first_image = image_batch[0]
# print(np.min(first_image),np.max(first_image))


#展示增强后图像变化
# plt.figure(figsize=(10,10))
# for images, _ in train_ds.take(1):
#     for i in range(9):
#         augmented_image = data_augmentation(images)
#         ax = plt.subplot(3, 3, i+1)
#         plt.imshow(augmented_image[1].numpy().astype("uint8"))
#         plt.axis("off")