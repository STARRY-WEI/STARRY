# -*- coding: utf-8 -*-
import numpy as np
import cv2
from matplotlib import pyplot as plt

# 读取图像并保存
# img = cv2.imread('image1.jpg',0)
# cv2.imshow('image',img)
# k = cv2.waitKey(0)
# if k == ord('a'): # wait for ESC key to exit
#     cv2.destroyAllWindows()
# elif k == ord('s'): # wait for 's' key to save and exit
#     cv2.imwrite('messigray.png',img)
#     cv2.destroyAllWindows()

# 使用matplotlib
# OpenCV follows BGR order, while matplotlib likely follows RGB order.
# img = cv2.imread('image1.jpg')
# b,g,r = cv2.split(img)
# img2 = cv2.merge([r,g,b])
# plt.subplot(121);plt.imshow(img) # expects distorted color
# plt.subplot(122);plt.imshow(img2) # expect true color
# plt.show()
# cv2.imshow('bgr image',img) # expects true color
# cv2.imshow('rgb image',img2) # expects distorted color


# openCV绘图
# img=np.zeros((200,200,3), np.uint8) 定义图片的大小
# Draw a diagonal blue line with thickness of 5 px
# a = cv2.rectangle(img,(0,50),(50,100),(255,0,0),10) #定义线条的起始位，以及颜色最后的tuple为元素，10为线条的粗细 ，(0,50)为左上角顶点，(50,100)为右下角顶点
# cv2.imshow('image',a)

# 获取并更改像素值
# img = cv2.imread('image1.jpg')
# resized = cv2.resize(img, (500, 300), ) # 500为横向大小，300为纵向大小,更改像素的大小
# corner = img[0:100,0:100] #读取像素块
# img[0:100,0:100] = (0,255,0) #更改读取的像素块
# cv2.imshow("Updated",img) #显示图像
# # cv2.imshow('image',resized)
# print img.shape #得到图片的大小，顺序为纵向，横向和通道大小
# print img.size  #可以返回图像的像素数目
# print img.dtype #返回的是图像的数据类型

# 图像 ROI ，抠图
# img = cv2.imread('image1.jpg')
# man = img[30:260,260:380] #数组为选取纵向的区间，和图片横向的区间，左上角为0,0，右下角为max，max
# img[30:260,0:120] = man
# cv2.imshow("Updated",img)

# 拆分合并图像通道
# img = cv2.imread('image1.jpg')
# img[:,:,0] = 0  # 更改通道0即B全部为0
# cv2.imshow("Updated",img)

# 填充边界，图像扩边
# BLUE=[255,0,0]
# img=cv2.imread('image1.jpg')
# b,g,r = cv2.split(img)
# img1 = cv2.merge([r,g,b])
# replicate = cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_REPLICATE)  #参数：输入图像， top, bottom, left, right， borderType 要添加那种类型的边界
# reflect = cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_REFLECT)
# reflect101 = cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_REFLECT_101)
# wrap = cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_WRAP)
# constant= cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_CONSTANT,value=BLUE)
# plt.subplot(231);plt.imshow(img1,'gray'),plt.title('ORIGINAL')  # subplot 表示创建一个2行，3列的图，p1为第一个子图，
# plt.subplot(232);plt.imshow(replicate,'gray'),plt.title('REPLICATE')
# plt.subplot(233);plt.imshow(reflect,'gray'),plt.title('REFLECT')
# plt.subplot(234);plt.imshow(reflect101,'gray'),plt.title('REFLECT_101')
# plt.subplot(235);plt.imshow(wrap,'gray'),plt.title('WRAP')
# plt.subplot(236);plt.imshow(constant,'gray'),plt.title('CONSTANT')
# plt.show()

#图像的加减法以及图像混合时图片的大小和类型要一致。
# img1 = cv2.imread('image1.jpg')
# img2 = cv2.imread('image2.jpg')
# resized = cv2.resize(img2, (650, 267), ) # 650为横向大小，267为纵向大小,更改像素的大小
# img3 = cv2.add(img1,resized) #单纯的图片相加
# dst=cv2.addWeighted(img1,0.9,resized,0.1,0) #对图片进行权重赋值
# cv2.imshow('a',dst)


# img1 = cv2.imread('image2.jpg')
# img2 = cv2.imread('image1.jpg')
# rows,cols,channels = img2.shape #得到图片的大小
# roi = img1[0:rows,0:cols] #找到感兴趣的地方
# img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)  #转换为灰度图，用cvtColor函数
# #threshold 指如果一个像素大于或小于阈值，重新给他分配一个值，第一参数是原始图片（为灰度模式图片）；第二个参数是阀值，
# # 用来分类像素,大于则赋予第三个参数的值；第三个参数指定大于（有时小于）阀值时，重新给它分配的值；第四个参数指定阀值模式,ret返回使用的阈值，mask返回的是最终选出来的矩阵
# ret, mask = cv2.threshold(img2gray, 100, 255, cv2.THRESH_BINARY) #变成binary的数组
# mask_inv = cv2.bitwise_not(mask) #bitwise表示按位取反
# img1_bg = cv2.bitwise_and(roi,roi,mask = mask)  #这里的roi感兴趣的地方，用于背景,mask是灰度图像，0是黑，255是白，也就是把白色部分的像素拿出来求与，意思就是选出白色和原图重叠的地方
# img2_fg = cv2.bitwise_and(img2,img2,mask = mask_inv)  #取roi中与mask_inv中不为0的值对应的像素的值，其他值为0，把logo中黑色部分提取出来，选出白色与原图重叠的地方
# dst = cv2.add(img1_bg,img2_fg) #将ROI中的logo和修改主要的图像
# img1[0:rows, 0:cols ] = dst   #替换原来的图像
# cv2.imshow('33',img1)


#追踪物体，并转化为hsv方式 在 OpenCV 的 HSV 格式中，H（色彩/色度）的取值范围是 [0，179]，S（饱和度）的取值范围 [0，255]，V（亮度）的取值范围 [0，255]
# img1 = cv2.imread('image2.jpg')
# lower_blue=np.array([50,50,50]) #设定蓝色的阈值
# upper_blue=np.array([150,255,255]) #设定蓝色的阈值
# hsv=cv2.cvtColor(img1,cv2.COLOR_BGR2HSV)
# mask=cv2.inRange(hsv,lower_blue,upper_blue) #根据阈值构建掩模
# res=cv2.bitwise_and(img1,img1,mask=mask)

# green=np.uint8([[[0,255,0]]])  #追踪对象的hsv值
# hsv_green=cv2.cvtColor(green,cv2.COLOR_BGR2HSV)  #追踪对象的hsv值
# print hsv_green #追踪对象的hsv值
# cv2.imshow('frame',img1)
# cv2.imshow('11',mask)
# cv2.imshow('22',res)

#图像的平移，旋转
# img = cv2.imread('image1.jpg')
# H = np.float32([[1,0,100],[0,1,50]])
# rows,cols = img.shape[:2]
# res = cv2.warpAffine(img,H,(cols,rows)) #需要图像、变换矩阵、变换后的大小
# plt.subplot(211);plt.imshow(img)
# plt.subplot(212);plt.imshow(res)
# plt.show()

#形态学转换
# img = cv2.imread('image1.jpg',0)
# kernel = np.ones((5,5),np.uint8)
# erosion = cv2.erode(img,kernel,iterations = 1)  #腐蚀
# dilation = cv2.dilate(img,kernel,iterations = 1)  #膨胀
# opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel) #开运算，通常用来去除噪声
# closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel) #闭运算，通常用来填充前景物体中的小洞，或者前景物体上的小黑点
# gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel) #前景物体的轮廓。
# tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel) #礼帽，原始图像与进行开运算之后得到的图像的差
# blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel) #黑帽，进行闭运算之后得到的图像与原始图像的差。
# cv2.getStructuringElement() #结构化元素，cv2.MORPH_RECT正方形，cv2.MORPH_ELLIPSE椭圆形，cv2.MORPH_CROSS圆形
# cv2.imshow('22',blackhat)

#边缘检测
# img = cv2.imread('image1.jpg',0)
# edges = cv2.Canny(img,50,255)
# plt.subplot(121);plt.imshow(img,cmap = 'gray')
# plt.title('Original Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(122);plt.imshow(edges,cmap = 'gray')
# plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
# plt.show()

# 角点检测
img = cv2.imread('image1.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# gray = np.float32(gray)
# dst = cv2.cornerHarris(gray,2,3,0.04)   # 第二个参数是角点检测中要考虑的领域大小，第三个参数是 Sobel 求导中使用的窗口大小，输入图像必须是 float32 ，最后一个参数在 0.04 到 0.06 之间
# dst = cv2.dilate(dst,None) #result is dilated for marking the corners, not important
# img[dst>0.01*dst.max()]=[0,0,255] # Threshold for an optimal value, it may vary depending on the image.

#找到最优的角点
# corners = cv2.goodFeaturesToTrack(gray,25,0.01,10)  #找到最佳的25个角点
# corners = np.int0(corners)
# for i in corners:
#     x,y = i.ravel()
#     cv2.circle(img,(x,y),3,255,-1)
# plt.imshow(img),plt.show()

# 图像匹配
# opencv教程p166

cv2.waitKey(0)
cv2.destroyAllWindows()