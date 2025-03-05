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
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix
import seaborn as sns


#kaggle数据集-5类-3690
data_dir = 'C:/Users/cherrytree/Desktop/biye_design/data/flower_photos'
save_path = 'C:/Users/cherrytree/Desktop/biye_design/machine_learning/my_flower/mobileV2_save_5'

class_names = ['daisy','dandelion','roses','sunflower','tulips']
#预处理参数
batch_size = 32
img_height = 224
img_width = 224
ratio = 0.2
#模型运行步数
epochs = 10

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

#生成热力图：

train_ds ,val_ds ,num_classes = get_data(img_height, img_width ,data_dir,ratio)

# #打乱数据集，配置数据集，保存在内存中 data.chche()
AUTOTUNE = tf.data.AUTOTUNE#？？？
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size = AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size = AUTOTUNE)

#迁移学习，MobileNetV2 https://gitee.com/song-laogou/Flower_tf2.3/blob/master/train_model.py
base_model = tf.keras.applications.MobileNetV2(input_shape=(img_height,img_width,3),
                                               include_top=False,weights='imagenet')

base_model.trainable = False
model = Sequential([
    layers.Rescaling(1./127.5,offset=-1,input_shape=(img_height,img_width,3)),
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(num_classes, activation='softmax')
])
model.summary()


#模型训练
model.compile(optimizer='adam' ,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
history = model.fit(train_ds,validation_data=val_ds,epochs=epochs)

#模型保存
model.save("models/moblieV2_5.h5")
model = tf.keras.models.load_model("models/moblieV2_5.h5")

# # 生成混淆矩阵
# true_labels = []
# predicted_labels = []
#
# for images, labels in val_ds:
#     predictions = model(images, training=False)
#     predicted_class = np.argmax(predictions, axis=1)
#     true_labels.extend(labels.numpy())
#     predicted_labels.extend(predicted_class)
#
# true_labels = np.array(true_labels)
# predicted_labels = np.array(predicted_labels)
#
# # 生成混淆矩阵
# cm = confusion_matrix(true_labels, predicted_labels)
#
# # 计算每行的总和
# cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#
# # 将混淆矩阵值转换为百分比
# cm_percentage = cm_normalized
#
# # 输出百分比形式的混淆矩阵
# print("Confusion Matrix (as percentages):")
# print(cm_percentage)
#
# # 可视化百分比形式的混淆矩阵
# plt.figure(figsize=(10, 8))
# sns.heatmap(cm_percentage, annot=True, fmt='.2f', cmap='Blues', xticklabels=val_ds.class_names, yticklabels=val_ds.class_names)
# plt.xlabel('Predicted Labels')
# plt.ylabel('True Labels')
# plt.title('Confusion Matrix (Percentages)')
# plt.xticks(rotation=90)  # 旋转X轴标签
# plt.show()

# #单个图像预测：
# probability_model = tf.keras.Sequential(
#     [model,tf.keras.layers.Softamx()]
# )
# predictions = probability_model.predict(val_ds)
# np.argmax(predictions[0])


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






