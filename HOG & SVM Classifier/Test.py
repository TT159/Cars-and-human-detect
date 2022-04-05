# from keras.models import load_model
# import numpy as np
# from keras.preprocessing import image
# from keras.preprocessing.image import load_img
# from PIL import ImageFilter
# from PIL import Image
# import cv2
# import os
# import keyboard
#
# PHOTO_SIZE = 120  #对图片大小进行限制，低于120KB的抛弃
# test_path = r'E:\样本图片 GroupImage\Group\Imageo'  # 目标
# weight_path = r'C:\code\model\CHKAnimal.h5' #权重文件
# outpath = r'C:\code\检测到的图片3'  #输出目录
# re_x = 64  #输入到神经网络中的文件的大小
#
#
# model = load_model(weight_path)
# files = os.listdir(test_path)
# os.chdir(test_path)
#
#

# img = Image.open(file)
# if img.mode != 'RGB': #如果不是RGB是其他的就转换成RGB
#     img = img.convert("RGB")
# img_origin = img
# img = img.resize((re_x,re_x), Image.ANTIALIAS) #缩放到事先指定的大小
# img = np.expand_dims(img, axis=0)
#
#
# predictions = model.predict(img)  #获取预测值
# if(not os.path.exists(outpath)):  #如果输出文件夹不存在就创建
#   os.makedirs(outpath)
# allpic += 1  #张数统计
# if(predictions[0][1]>predictions[0][0] #分类1（是动漫图片）的可能性更大并且文件大小超过设置值
# and os.path.getsize(file)>PHOTO_SIZE*1024):
#  print(file,"predict:", predictions)  #打印出预测值
#  img_origin.save(os.path.join(outpath,file)) #保存文件
#  os.remove(file)  #删除原先的图片
#
#
#

import numpy as np
from sklearn import svm

# x_data = ...  # 写自己的
# y_label = ...  # 写自己的


# xlf = svm.SVC(C=1,gamma=1,kernel='rbf',probability=True) # 构建模型
# xlf.fit(x_data,y_label) # 训练模型
from sklearn.externals import joblib

joblib.dump(xlf, 'modle.pkl')  # 储存模型 .dump(模型，‘模型名')

x_test = ...  # 写自己的
xlf_lo = joblib.load('modle.pkl')  # 读取模型
y_test = xlf_lo.predict(x_test) # 模型使用


