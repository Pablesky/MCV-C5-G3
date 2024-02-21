import torch
from torchvision import datasets
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from torch import nn
from torchinfo import summary



def load_data(train_dir, test_dir, batch_size=8):
    transformTrain = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.uint8, scale=True),
        v2.RandomAffine(
            degrees=0,
            translate=(0.2, 0.0),
            scale=(1.0, 1.0),
            shear=0.2
        ),
        v2.ColorJitter(brightness=(0.7, 1.3)),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ToDtype(torch.float32, scale=True)
    ])

    transformTest = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
    
    train_data = datasets.ImageFolder(root=train_dir, # target folder of images
                                  transform=transformTrain, # transforms to perform on data (images)
                                  target_transform=None) # transforms to perform on labels (if necessary)

    test_data = datasets.ImageFolder(root=test_dir, 
                                    transform=transformTest)
    
    train_dataloader = DataLoader(dataset=train_data, 
                              batch_size=batch_size, # how many samples per batch?
                              num_workers=1, # how many subprocesses to use for data loading? (higher = more)
                              shuffle=True) # shuffle the data?

    test_dataloader = DataLoader(dataset=test_data, 
                             batch_size=1, 
                             num_workers=1, 
                             shuffle=False) # don't usually need to shuffle testing data

    return train_dataloader, test_dataloader

class start(nn.Module):
    def __init__(self):
        super(start, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=(3, 3), padding='same')
        self.batchNorm1 = nn.BatchNorm2d(self.conv1.out_channels)
        self.activate_layer = nn.LeakyReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchNorm1(x)
        x = self.activate_layer(x)

        return x

class separable_block(nn.Module):
    def __init__(self, input_channels, output_channels, padding):
        super(separable_block, self).__init__()

        self.conv1 = nn.Conv2d(input_channels, input_channels, kernel_size=(3, 3), padding=padding)
        self.batchNorm1 = nn.BatchNorm2d(self.conv1.out_channels)
        self.activate_layer1 = nn.LeakyReLU()

        self.conv2 = nn.Conv2d(input_channels, output_channels, kernel_size=(1, 1), padding=padding)
        self.batchNorm2 = nn.BatchNorm2d(self.conv2.out_channels)
        self.activate_layer2 = nn.LeakyReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchNorm1(x)
        x = self.activate_layer1(x)

        x = self.conv2(x)
        x = self.batchNorm2(x)
        x = self.activate_layer2(x)

        return x
    
class best_model(nn.Module):
    def __init__(self):
        super(best_model, self).__init__()

        self.start1 = start()
        self.separable_block1 = separable_block(16, 32, 'same')
        self.separable_block2 = separable_block(32, 64, 'same')
        self.separable_block3 = separable_block(64, 32, 'same')
        self.separable_block4 = separable_block(32, 64, 'same')
        self.separable_block5 = separable_block(64, 128, 'same')
        self.separable_block6 = separable_block(128, 32, 'same')
        self.separable_block7 = separable_block(32, 128, 'same')

        self.fc = nn.Linear(128, 8)

        self.softmax_layer = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.start1(x)
        x = self.separable_block1(x)
        x = self.separable_block2(x)
        x = self.separable_block3(x)
        x = self.separable_block4(x)
        x = self.separable_block5(x)
        x = self.separable_block6(x)
        x = self.separable_block7(x)

        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))

        x = torch.flatten(x, start_dim=1)

        x = self.fc(x)

        x = self.softmax_layer(x)

        return x
    
def accuracy_fn(y_true, y_pred):
    correct = int(y_true == y_pred)
    return correct

def accuracy_fn_tensors(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item() # torch.eq() calculates where two tensors are equal
    acc = (correct / len(y_pred)) * 100 
    return acc

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    root_dir = '../MIT_small_train_1/'
    train_dir = root_dir + 'train'
    test_dir = root_dir + 'test'

    train_dataloader, test_dataloader = load_data(train_dir, test_dir)

    model = best_model()

    model = model.to(device)

    optimizer = torch.optim.NAdam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.CrossEntropyLoss()

    epochs = 200

    loss_train = 0
    for epoch in range(epochs):
        model.train()

        for i, data in enumerate(train_dataloader):
            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)

            preds = outputs.argmax(dim=1)

            acc = accuracy_fn_tensors(labels, preds)

            loss = loss_fn(outputs, labels)
            loss_train += loss
            loss.backward()
            optimizer.step()

        loss_train_1 = loss_train / len(train_dataloader)
        acc_1 = acc / len(train_dataloader)

        model.eval()
        test_loss = 0
        accuracy_test = 0
        with torch.inference_mode():
            for i, data in enumerate(test_dataloader):

                inputs, labels = data

                # 1. Forward pass
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                test_logits = model(inputs)
                # 2. Calculate test loss and accuracy
                test_loss += loss_fn(test_logits, labels)

                test_pred = torch.argmax(test_logits)
                
                accuracy_test += accuracy_fn(labels[0], test_pred)

        test_loss_1 = test_loss / len(test_dataloader)
        accuracy_test_1 = accuracy_test / len(test_dataloader)

            # Print out what's happening
        if epoch % 10 == 0:
            print(f"Epoch: {epoch} | Loss: {loss_train_1:.5f}, Train Accuracy: {acc_1:.5f}, Test Loss: {test_loss_1:.5f}, Test Accuracy: {accuracy_test_1:.5f}")

    


