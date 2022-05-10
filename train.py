from CNN import CNN
from data import BoWFireDataset
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader
from torchvision import transforms

if __name__ == '__main__':
    batch_size = 8
    transform = transforms.Compose([transforms.ToPILImage(),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor()
                                    ])
    train_dataset = BoWFireDataset(transform=transform)
    test_dataset = BoWFireDataset(transform=transforms.ToTensor())
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    model = CNN()
    optimizer = Adam(model.parameters(), lr=5e-4, weight_decay=0.1)#,  momentum=0.9)#)#)
    loss_fn = CrossEntropyLoss()
    all_epoch = 100

    for current_epoch in range(all_epoch):
        model.train()
        for idx, (train_x, train_label) in enumerate(train_loader):
            optimizer.zero_grad()
            predict_y = model(train_x.float())
            loss = loss_fn(predict_y, train_label.long())
            loss.backward()
            optimizer.step()

        all_correct_num = 0
        all_sample_num = 0
        all_loss = 0
        model.eval()
        for idx, (test_x, test_label) in enumerate(test_loader):
            predict_y = model(test_x.float()).detach()
            predict_label = np.argmax(predict_y, axis=-1)
            current_correct_num = predict_label == test_label
            all_correct_num += np.sum(current_correct_num.numpy(), axis=-1)
            all_sample_num += current_correct_num.shape[0]
            all_loss += loss_fn(predict_y, test_label.long()).sum()
        acc = all_correct_num / all_sample_num
        print('[{}/{}] loss: {:.2f}  accuracy: {:.2f}'.format(current_epoch+1, all_epoch, all_loss, acc))
        torch.save(model, 'models/fire_{:.2f}.pkl'.format(acc))