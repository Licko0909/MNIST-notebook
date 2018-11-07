# MNIST-notebook

## 目录Content：

+ 1、自建MNIST数据集，批量读取手写体，数据预处理与对接
+ 2、建立MNIST_CNN模型，并识别预测
+ 3、混淆矩阵可视化
+ 4、错误识别集提取可视化
+ 5、保存MNIST_CNN模型

```python

#读取模块，设定随机种子
import os
from PIL import Image
import random
import scipy
import time
from keras.datasets import mnist
from keras.utils import np_utils
import random
np.random.seed(10)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
%matplotlib inline

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
```

```python

'''
===========    第一部分：自建MNIST数据集，批量读取手写体，数据预处理与对接    =========
'''

def readData(path,isShuffle):
    '''
    path 为图片所在的路径
    isShuffle 为是否随机打乱
    '''
    # 读取路径下的所有图片
    files = os.listdir(path)    
 
    #是否需要打乱顺序读取图片
    if isShuffle == True:     
        random.shuffle(files)
    dataLen = len(files)

    # 设置占位符
    images = np.empty((dataLen,28,28),dtype=np.uint8)
    label = np.empty((dataLen,),dtype='uint8')

    # 遍历目录下的所有图片
    for i in range(dataLen):
        file = files[i]
        # 读取图片信息
        arr = scipy.misc.imread(path + file,flatten=True) #需要配套数值变换
        images[i,:,:] = arr
        # 对文件名进行两次拆分，取第二次拆分后的结果作为图片的 label
        label[i] = int(file.split('_')[1].split('.')[0]) 
        
    return (images,label)


path = 'D:/myP/textbook/手写体实践/数字数据集2/' #图片所在的路径，记得路径末尾添加反斜杠
#path = 'D:/pweave_x/picture/'
#生成测试集
(images_Test,label_Test) = readData(path,True)     #False按顺序，True为乱序
#(images_Train,label_Train) = readData(path,False)

print("个数、长、宽分别为:",images_Test.shape)       #查看shape



#绘制0-9各数字的数量图
import seaborn as sns
count = sns.countplot(label_Test)


#显示单张图片（第553张）
plt.figure()
plt.imshow(images_Test[553],cmap='gray')  #没有cmap='gray'会显示原色（黄色）
plt.axis('off')
plt.show()


#数据预处理、标准化
img_data=255.0-images_Test.reshape(images_Test.shape[0],784).astype('float32')
img_data=(img_data/255.0*0.99)+0.01  #1行784列，压缩到



#加载MNIST的数据集
#若不想更新，想一直保存每次测试的数字，就不要重新加载数据集
(x_train_image, y_train_label), (x_test_image, y_test_label) = mnist.load_data()

x_Train =x_train_image.reshape(60000, 784).astype('float32')
x_Test = x_test_image.reshape(10000, 784).astype('float32')

x_Train_normalize = x_Train/ 255
x_Test_normalize = x_Test/ 255


#reshape成n*28*28*1（针对MNIST数据集）
x_Train4D=x_Train.reshape(x_Train.shape[0],28,28,1).astype('float32')
x_Test4D=x_Test.reshape(x_Test.shape[0],28,28,1).astype('float32')


#数据标准化
x_Train4D_normalize = x_Train4D / 255
x_Test4D_normalize1 = x_Test4D / 255


#热编码
y_TrainOneHot = np_utils.to_categorical(y_train_label)
y_TestOneHot = np_utils.to_categorical(y_test_label)
y_TestOneHot01 =np_utils.to_categorical(label_Test)    #为了热编码对接


#reshape成n*28*28*1（针对自建数据集）
img_array4D=img_data.reshape(img_data.shape[0],28,28,1).astype('float32')


#自建数据集+MNIST数据集对接，①测试集对接，②热编码对接
#注意：需要前后的数据范围符合，同时数据类型和维度一样,范围周围0，数字为0-1.
x_Test4D_normalize = np.row_stack((x_Test4D_normalize1, img_array4D)) #测试集对接
y_TestOneHot= np.row_stack((y_TestOneHot, y_TestOneHot01))            #热编码对接
#label_Test_all= np.row_stack((y_test_label,label_Test))              #label对接(接不上)


#分割出验证集 = 0.05的训练集当作验证集（6月8日晨模型）
random_seed = 2
X_train1, X_val1, Y_train1, Y_val1 = train_test_split(
        x_Train4D_normalize, y_TrainOneHot, 
        test_size = 0.05, random_state=random_seed)
```


```python

'''
=========   第二部分：建立MNIST_CNN模型，并预测   =============
'''

#建立模型
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D


#设定空的模型框架
model = Sequential()


#建立卷积层1
""" 
建立卷积层1 filters=16: 16个滤镜 
kernel_size=(5,5)： 每个滤镜5x5大小 
padding='same'：卷积运算所产生的卷积图像大小不变 
input_shape=(28,28,1)：图像的形状为28x28，图像是单色灰度图像，所以用1表示，                                      
如果图像是彩色图像RGB，所以用3表示， 
activation='relu'：激活函数 （x小于0，y=0；x大于等于0，y=x） 
""" 
model.add(Conv2D(filters=16,
                 kernel_size=(5,5),
                 padding='same',
                 input_shape=(28,28,1), 
                 activation='relu'))



#建立池化层1
""" 
建立池化层1 参数pool_size=(2, 2)，
执行第1次缩减采样，将16个28x28的图像缩小为16个14x14 的图像 
""" 
model.add(MaxPooling2D(pool_size=(2, 2)))



#建立卷积层2 
""" 
建立卷积层2 
执行第2次卷积运算，将原来的16个滤镜变为36个滤镜， 
input_shape=14x14的36个图像
 """  
model.add(Conv2D(filters=36,
                 kernel_size=(5,5),
                 padding='same',
                 activation='relu'))


#建立池化层2 
""" 
建立池化层2 
参数pool_size=(2, 2)，执行第2次缩减采样，
将36个14x14的图像缩小为36 个7x7的图像
""" 
model.add(MaxPooling2D(pool_size=(2, 2)))


#建立Dropout层
""" 
建立Dropout层，可以防止过拟合，
设置25%的随机丢弃率.
""" 
model.add(Dropout(0.25))


#建立平坦层
"""  
平坦层 36*7*7 = 1764个神经元 

隐含层 1764*128+128 = 225920 个神经元

Dropout层 128个神经元，50%丢弃率

输出层 10*128+10 = 1290个神经元

""" 
model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(10,activation='softmax'))

#输出模型摘要
print(model.summary())




# 定义优化器
optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

# 编译模型
model.compile(optimizer = optimizer ,
              loss = "categorical_crossentropy",
              metrics=["accuracy"])


# 设置学习速率退火机
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)


epochs = 10 # Turn epochs to 10 to get 0.999 accuracy
batch_size = 186  #大样本用一个适当值，小样本设置全批次
#理论在这，https://blog.csdn.net/qq_20259459/article/details/53943413



#---3.3 数据扩充\数据增强
#只设置了四种数据增强类型，随机旋转、放缩、水平宽度位移、水平高度位移

datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images

datagen.fit(x_Test4D_normalize)


# 模型训练 (need 5min)  用"训练集"训练，用"验证集"调教，得到一个识别模型，再用测试集测试
history = model.fit_generator(datagen.flow(X_train1,Y_train1, batch_size=batch_size),
                              epochs = epochs, validation_data = (X_val1,Y_val1),
                              verbose = 2, steps_per_epoch=x_Train.shape[0] // batch_size
                              , callbacks=[learning_rate_reduction])

# 绘制训练和验证的损耗和精度曲线
fig, ax = plt.subplots(2,1)
ax[0].plot(history.history['loss'], color='b', label="Training loss")
ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(history.history['acc'], color='b', label="Training accuracy")
ax[1].plot(history.history['val_acc'], color='r',label="Validation accuracy")
legend = ax[1].legend(loc='best', shadow=True)


#验证集中的准确率(99.0667%)
scores_val = model.evaluate(X_val1,Y_val1)
print("验证集中的准确率：",scores_val[1])

#620张测试集中的准确率(99.8387%)
scores_620 = model.evaluate(img_array4D, y_TestOneHot01) 
print("620张测试集的准确率：",scores_620[1])

#10620张测试集中的准确率(99.9246%)
scores_all = model.evaluate(x_Test4D_normalize, y_TestOneHot) 
print("10620张测试集的准确率：",scores_all[1])


#预测
prediction=model.predict_classes(x_Test4D_normalize)
prediction620=prediction[10000:11000]
#print ('手写体的预测值:',prediction)  #设置1000的长度
#print ('手写体的标签为:',label_Test)  #label



#查看预测结果，可视化(查看620张中的前50张，最好能够随机，待解决)
def plot_images_labels_prediction(images,labels,prediction,idx,num=50):
    fig = plt.gcf()
    fig.set_size_inches(12, 18)    #间隔的宽与高
    if num>50: num=50
    for i in range(0, num):
        ax=plt.subplot(10,5, 1+i)  #设置按10行5列展示
        ax.imshow(images[idx], cmap='binary')

        ax.set_title("label=" +str(labels[idx])+
                     ",predict="+str(prediction[idx])
                     ,fontsize=10) 
        
        ax.set_xticks([]);ax.set_yticks([])        
        idx+=1 
    plt.show()

plot_images_labels_prediction(images_Test,label_Test,prediction620,idx=0)
```
