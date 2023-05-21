# 导入必要的模块
import cv2 # 用于图像处理和摄像头操作
import torch # 用于深度学习
import numpy as np # 用于数组操作
from torchvision import transforms

# 加载人脸检测模型，使用OpenCV自带的Haar特征级联分类器
#face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 加载年龄和性别预测模型，使用之前训练好的PyTorch模型
model = torch.load('age_gender_model_4.pt',map_location="cpu")
# 将模型设置为评估模式，关闭梯度计算和Dropout层
model.eval()
# 判断是否有GPU可用，如果有则将模型移动到GPU上
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 定义年龄和性别的类别标签
gender_labels = ['Male', 'Female']

# 打开电脑本地摄像头，使用cv2.VideoCapture类，传入0表示默认摄像头
cap = cv2.VideoCapture(0)

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                    std=[0.5, 0.5, 0.5]),
                                transforms.Resize((64, 64))
                                ])

# 循环读取摄像头的每一帧图像
while True:
    # 从摄像头读取一帧图像，返回一个布尔值和一个数组
    ret, frame = cap.read()
    # 如果读取成功，则继续处理图像
    if ret:
        # 将图像转换为灰度图，便于人脸检测
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 使用人脸检测模型在灰度图上检测人脸，返回一个列表，每个元素是一个包含人脸位置信息的元组
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        # 循环遍历每一个检测到
                # 循环遍历每一个检测到的人脸
        for (x, y, w, h) in faces:
            # 在原始图像上绘制一个矩形框，表示人脸的位置
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            # 从原始图像中裁剪出人脸区域，作为模型的输入
            face = frame[y:y+h, x:x+w]
            # 将人脸区域调整为64x64的大小，与模型训练时的输入一致
            # face = cv2.resize(face, (64, 64))
            # 将图像转换为PyTorch张量，并增加一个批次维度
            # face = torch.from_numpy(face).permute(2, 0, 1).unsqueeze(0).float()
            # 将图像张量移动到相同的设备上
            face = transform(face).unsqueeze(0).float()
            face = face.to(device)
            # 使用模型进行预测，得到年龄和性别的输出
            #print(face)
            age_out, gender_out = model(face)
            print('模型输出年龄',age_out)
            age_out = age_out.squeeze(0)
            print('age',age_out)
            # 将年龄输出转换为标量，并乘以116，得到年龄值的估计
            #age = age_out.item()
            age = int(round(age_out.item() * 116))
            # 将性别输出转换为概率分布，并取最大值的索引，得到性别类别的预测
            gender = torch.softmax(gender_out, dim=1).argmax().item()
            # 根据年龄值和性别类别，得到对应的标签文本
            gender_label = gender_labels[gender]
            # 在原始图像上绘制标签文本，显示在人脸框的上方
            cv2.putText(frame, f'{age}, {gender_label}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        # 显示处理后的图像，使用cv2.imshow函数，传入窗口名称和图像数组
        cv2.imshow('Age and Gender Detection', frame)
        # 等待用户按键，如果按下Esc键，则退出循环
        key = cv2.waitKey(1)
        if key == 27:
            break
    # 如果读取失败，则打印错误信息并退出循环
    else:
        print('Failed to read frame')
        break

# 释放摄像头资源，使用cv2.VideoCapture类的release方法
cap.release()
# 关闭所有窗口，使用cv2.destroyAllWindows函数
cv2.destroyAllWindows()
