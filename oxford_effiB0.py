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
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import EfficientNetB0

#kaggle数据集-5类-3690
data_dir = 'C:/Users/cherrytree/Desktop/biye_design/data/flower_photos'

#预处理参数
batch_size = 64
img_height = 224
img_width = 224
ratio = 0.2
epochs = 5
num_classes = 5
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

    return train_ds, val_ds,num_classes

train_ds,val_ds, num_classes = get_data(img_height, img_width ,data_dir,ratio)

#数据增强
def augment_data(train_ds):
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    for images, labels in train_ds:
        X_train.append(images.numpy())
        y_train.append(labels.numpy())
    X_train = np.concatenate(X_train,axis = 0)
    y_train = np.concatenate(y_train, axis=0)
    for images, labels in val_ds:
        X_test.append(images.numpy())
        y_test.append(labels.numpy())
    X_test = np.concatenate(X_test, axis=0)
    y_test = np.concatenate(y_test, axis=0)
    # 定义数据增强
    train_datagen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True
    )

    val_datagen = ImageDataGenerator()
    train_generator = train_datagen.flow(X_train,y_train,batch_size = batch_size,shuffle = True)
    val_generator = val_datagen.flow(X_test,y_test,batch_size = batch_size,shuffle = True)
    return train_generator,val_generator

train_generator, val_generator = augment_data(train_ds)
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

# history = model.fit(train_generator,validation_data=train_generator,epochs=epochs)
#
# model.save("models/effiB0_5.h5")
model = tf.keras.models.load_model("models/effiB0_5.h5")

loss, accuracy = model.evaluate(val_generator)
print('Test accuracy :', accuracy)

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
sns.heatmap(cm_percentage, annot=True, fmt='.2f', cmap='Oranges', xticklabels=val_ds.class_names, yticklabels=val_ds.class_names)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix (Percentages)')
plt.xticks(rotation=90)  # 旋转X轴标签
plt.show()

# #呈现训练结果
# acc = history.history['accuracy']
# val_acc = history.history['val_accuracy']
# loss = history.history['loss']
# val_loss = history.history['val_loss']
#
# epochs_range = range(epochs)
#
# plt.figure(figsize=(8,8))
# plt.subplot(1, 2, 1)
# plt.plot(epochs_range, acc, label='Training Accuracy')
# plt.plot(epochs_range, val_acc, label='Validation Accuracy')
# plt.legend(loc='lower right')
# plt.title('Training and Validation Accuracy')
#
# plt.subplot(1, 2, 2)
# plt.plot(epochs_range, loss, label='Training Loss')
# plt.plot(epochs_range, val_loss, label='Validation Loss')
# plt.legend(loc='upper right')
# plt.title('Training and Validation Loss')
# plt.show()









