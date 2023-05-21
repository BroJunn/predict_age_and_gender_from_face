import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import AgeGenderDataset

train_on_gpu = torch.cuda.is_available()
# 加载保存的模型
model = torch.load('age_gender_model_4.pt')
# 设置模型为评估模式
model.eval()
# 创建一个测试数据集类的实例，这个类可以从指定的文件夹中读取人脸图片和标签
test_ds = AgeGenderDataset("./UTK/UTKFace_test")
# 设置每个批次的样本数
bs = 4
# 创建一个数据加载器，可以从数据集中顺序抽取批次的数据
test_dataloader = DataLoader(test_ds, batch_size=bs, shuffle=False)
# 初始化一个变量，用于累计测试损失
test_loss = 0.0
# 初始化两个变量，用于记录年龄和性别的准确率
age_error = 0.0
gender_acc = 0.0
# 损失函数
mse_loss = torch.nn.MSELoss()
cross_loss = torch.nn.CrossEntropyLoss()

index = 0
# 对每一个批次进行循环
for i_batch, sample_batched in tqdm(enumerate(test_dataloader)):
    with torch.no_grad():
        # 从批次中获取图片和标签
        images_batch, age_batch, gender_batch = \
            sample_batched['image'], sample_batched['age'], sample_batched['gender']
        # 如果有GPU可用，就把图片和标签放到GPU上
        if train_on_gpu:
            model = model.cuda()
            images_batch, age_batch, gender_batch = images_batch.cuda(), age_batch.cuda(), gender_batch.cuda()
        # 前向传播，得到模型的输出
        m_age_out_, m_gender_out_ = model(images_batch)
        # 把年龄标签转换为浮点数，并调整形状为一维向量
        age_batch = age_batch.view(-1, 1).float()
        # 把性别标签转换为整数，并调整形状为一维向量
        gender_batch = gender_batch.long()
        # 计算批次的总损失，等于年龄损失和性别损失之和
        loss = mse_loss(m_age_out_, age_batch) + cross_loss(m_gender_out_, gender_batch)
        # 累加测试损失
        test_loss += loss.item()
        # 计算年龄预测的平均绝对误差，并累加到年龄准确率中
        age_mae = torch.mean(torch.abs(m_age_out_ - age_batch))
        age_error += age_mae
        # age_acc += (1 - age_mae / 100)
        # 计算性别预测的准确率，并累加到性别准确率中
        gender_pred = torch.argmax(m_gender_out_, dim=1)
        gender_correct = torch.sum(gender_pred == gender_batch)
        gender_acc += gender_correct / bs

# 计算平均损失，等于总损失除以样本数
test_loss = test_loss / test_ds.__len__()
# 计算年龄和性别的平均准确率，等于总准确率除以批次数
age_error = age_error / len(test_dataloader)
gender_acc = gender_acc / len(test_dataloader)

# 显示测试集的损失值和准确率值
print('Test Loss: {:.6f} '.format(test_loss))
print('Age Mean Error: {:.2f}'.format(age_error * 106))
print('Gender Accuracy: {:.2f}% '.format(gender_acc * 100))
