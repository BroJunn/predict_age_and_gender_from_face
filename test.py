import torch
from torchvision import transforms
import cv2 as cv


# 这一行定义了一个函数，用于将图片转换为张量，并进行归一化和裁剪等预处理操作
def transform_image(image):
    # 这一行创建了一个转换器，包含了多个转换操作
    transformer = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                                 std=[0.5, 0.5, 0.5]),
                                             transforms.Resize((64, 64))
                                             ])
    # 这一行返回转换后的图片张量
    return transformer(image)


gender_labels = ['Male', 'Female']
# 这一行定义了一个函数，用于对模型的输出进行解释，得到年龄和性别的预测值
def interpret_output(output):
    # 这一行将输出分解为年龄和性别两个变量
    age_out, gender_out = output
    print(age_out)
    print(gender_out)
    # 这一行将年龄张量转换为标量，并四舍五入为整数
    age_pred = int(round(age_out.item()*116))
    # 这一行将性别张量转换为标量，并根据最大值的索引判断为男或女
    #gender_pred = "male" if gender_out.argmax() == 1 else "female"
    # 将性别输出转换为概率分布，并取最大值的索引，得到性别类别的预测
    gender = torch.softmax(gender_out, dim=1).argmax().item()
    # 根据年龄值和性别类别，得到对应的标签文本
    #age_label = age_labels[int(age // 10)]
    gender_label = gender_labels[gender]
    # 这一行返回年龄和性别的预测值
    return age_pred, gender_label


# 这一行打开图片文件，文件名为'image.jpg'
image = cv.imread("./test/test5.jpg")
# 这一行调用transform_image函数，对图片进行预处理
image_tensor = transform_image(image)
# 这一行增加一个维度，使图片张量符合模型的输入要求
image_tensor = image_tensor.unsqueeze(0)
# 这一行加载模型文件，文件名为'age_gender_model.pt'
model = torch.load("age_gender_model_4.pt",map_location="cpu")
# 这一行将模型设置为评估模式，关闭梯度计算和随机性
model.eval()
# 这一行进行前向传播，将图片张量输入模型，得到输出张量
output_tensor = model(image_tensor)
# 这一行调用interpret_output函数，对输出张量进行解释，得到年龄和性别的预测值
age_pred, gender_pred = interpret_output(output_tensor)



# 这一行打印出预测结果
print(f"The predicted age is {age_pred} and the predicted gender is {gender_pred}.")