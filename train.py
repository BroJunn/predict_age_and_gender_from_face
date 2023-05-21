import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import MyMulitpleTaskNet
from dataset import AgeGenderDataset


if __name__ == "__main__":
    # 检查是否可以利用GPU
    train_on_gpu = torch.cuda.is_available()
    if not train_on_gpu:
        print('CUDA is not available.')
    else:
        print('CUDA is available!')

    model = MyMulitpleTaskNet()
    # print(model)

    # 使用GPU
    if train_on_gpu:
        model.cuda()

    ds_train = AgeGenderDataset("./UTK/UTKFace_train")
    ds_val = AgeGenderDataset("./UTK/UTKFace_test")
    num_train_samples = ds_train.__len__()
    num_val_samples = ds_val.__len__()
    
    bs = 16
    dataloader_train = DataLoader(ds_train, batch_size=bs, shuffle=True)
    dataloader_val = DataLoader(ds_val, batch_size=bs, shuffle=False)
   
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
   

    # 损失函数
    mse_loss = torch.nn.MSELoss()
    cross_loss = torch.nn.CrossEntropyLoss()
    
    index = 0

    num_epochs = 25
    val_loss_best = 100000000
    for epoch in range(num_epochs):
        # train
        model.train()
        train_loss = 0.0
          # 依次取出每一个图片与label
        for i_batch, sample_batched in tqdm(enumerate(dataloader_train)):
            images_batch, age_batch, gender_batch = \
                sample_batched['image'], sample_batched['age'], sample_batched['gender']
            if train_on_gpu:
                images_batch, age_batch, gender_batch = images_batch.cuda(), age_batch.cuda(), gender_batch.cuda()
                
            optimizer.zero_grad()

            # forward pass
            m_age_out_, m_gender_out_ = model(images_batch)
            # age_batch = age_batch.view(-1, 1).float()
            # gender_batch = gender_batch.long()
            age_batch = age_batch.view(-1, 1).float()
            gender_batch = gender_batch.long()

            # calculate the batch loss
            loss = mse_loss(m_age_out_, age_batch) + cross_loss(m_gender_out_, gender_batch)

            # backward pass
            loss.backward()

            # perform a single optimization step (parameter update)
            optimizer.step()

            # update training loss
            train_loss += loss.item()
            if index % 20 == 0:
                print('step: {} \tTraining Loss: {:.6f} '.format(index, loss.item()))
            index += 1

        # 计算平均损失
        train_loss = train_loss / num_train_samples
        
        
        # evaluation
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for i_batch, sample_batched in tqdm(enumerate(dataloader_val)):
                images_batch, age_batch, gender_batch = \
                    sample_batched['image'], sample_batched['age'], sample_batched['gender']
                if train_on_gpu:
                    images_batch, age_batch, gender_batch = images_batch.cuda(), age_batch.cuda(), gender_batch.cuda()
                
                
                age_batch = age_batch.view(-1, 1).float()
                gender_batch = gender_batch.long()
                m_age_out_, m_gender_out_ = model(images_batch)
                age_batch = age_batch.view(-1, 1)
                
                loss = mse_loss(m_age_out_, age_batch) + cross_loss(m_gender_out_, gender_batch)

                val_loss += loss.item()
                
            val_loss = val_loss / num_val_samples

            if val_loss < val_loss_best:
                val_loss_best = val_loss
                torch.save(model, 'age_gender_model_' + str(epoch) +'.pt')
        # 显示训练集与验证集的损失函数
        print('Epoch: {} \tTraining Loss: {:.6f}, Validation Loss: {:.6f} '.format(epoch, train_loss, val_loss))

    # save model
    torch.save(model, 'age_gender_model_last.pt')

