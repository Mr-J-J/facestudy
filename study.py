# 调用摄像头显示图像
# import numpy as np
# import cv2
# cap = cv2.VideoCapture(0)#从计算机上的第一个网络摄像头返回视频
# while True:
#     ret, frame = cap.read()
#     #其中ret是布尔值，如果读取帧是正确的则返回True，如果文件读取到结尾，它的返回值就为False。
#     #frame就是每一帧的图像，是个三维矩阵。
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     cv2.imshow('frame', gray)
#     if cv2.waitKey(0) & 0xFF == ord('q'):
#         break
# cap.release()
# cv2.destroyAllWindows()
# #cap.read()返回的是一个布尔值和一个图像，如果布尔值为True，则说明读取到了一帧图像，否则说明读取完毕。

# 调用摄像头并且输出视频
# import cv2
# cap = cv2.VideoCapture(0)
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))
# while True:
#     ret, frame = cap.read()
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     out.write(frame)
#     cv2.imshow('frame',gray)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# cap.release()
# out.release()
# cv2.destroyAllWindows()

# 画线
# import numpy as np
# import cv2
# img = cv2.imread('data/User.jiang.2.jpg', cv2.IMREAD_COLOR)
# pts = np.array([[10,5],[200,300],[200,400],[100,400],[50,10]], np.int32)
# pts = pts.reshape((-1,1,2))
# cv2.polylines(img, [pts], True, (0,255,255), 5)
# cv2.imshow('img', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# # 文字
# import numpy as np
# import cv2
# img = cv2.imread('data/User.jiang.2.jpg',cv2.IMREAD_COLOR)
# font = cv2.FONT_HERSHEY_SIMPLEX
# cv2.putText(img, 'OpenCV Tuts!',(10,400), font, 2, (200,255,155), 5, cv2.LINE_AA)
# #参数依次为 图片，文字，文字起始位置，font,文字大小，文字颜色，文字粗细，cv2.LINE_AA
# cv2.imshow('image',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
#

# # 图像模糊
# import numpy as np
# import cv2
# img = cv2.imread('data/User.jiang.2.jpg',cv2.IMREAD_COLOR)
# blur = cv2.blur(img,(5,5))
# cv2.imshow('image',img)
# cv2.imshow('blur',blur)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # 阈值化，少于多少亮度或高于多少亮度的都设为黑色
# import cv2
# import numpy as np
# img = cv2.imread('data/User.jiang.2.jpg')
# grayscaled = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# retval, threshold = cv2.threshold(grayscaled, 3, 255, cv2.THRESH_BINARY)
# cv2.imshow('original',img)
# cv2.imshow('threshold',threshold)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # 图像添加文字
# import cv2
# # 读图片
# img = cv2.imread('data/User.jiang.2.jpg')
# # 图片信息
# print('图片尺寸:', img.shape)
# print('图片数据:', type(img), img)
# # 显示图片
# cv2.imshow('pic title', img)
# cv2.waitKey(0)
# # 添加文字
# cv2.putText(img, 'Learn Python with Crossin', (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 4)
# # 保存图片
# cv2.imwrite('data/User.jpg', img)

# # 常见的图像处理函数
# import cv2
# import numpy as np
# img = cv2.imread('data/User.jiang.2.jpg')
# # 灰度图
# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.imshow('img/Lenna_gray.png', img_gray)
# # 二值化
# _, img_bin = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
# cv2.imshow('img/Lenna_bin.png', img_bin)
# # 平滑
# img_blur = cv2.blur(img, (5, 5))
# cv2.imshow('img/Lenna_blur.png', img_blur)
# # 边缘提取
# img_edge = cv2.Canny(img, 100, 200)
# cv2.imshow('img/Lenna_edge.png', img_edge)
# # 水平翻转
# img_flip = cv2.flip(img, 1)
# cv2.imshow('img/Lenna_flip.png', img_flip)
# # 垂直翻转
# img_flip = cv2.flip(img, 0)
# cv2.imshow('img/Lenna_flip.png', img_flip)
# # 旋转
# img_rotate = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
# cv2.imshow('img/Lenna_rotate.png', img_rotate)
# # 图像缩放
# img_resize = cv2.resize(img, (int(img.shape[1]/2), int(img.shape[0]/2)))
# cv2.imshow('img/Lenna_resize.png', img_resize)
# # 图像裁剪
# img_crop = img[100:300, 100:300]
# cv2.imshow('img/Lenna_crop.png', img_crop)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

