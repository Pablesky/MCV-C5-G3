import torch
from torchvision.transforms import v2
from torchvision import datasets
from torch.utils.data import DataLoader

def load_data(train_dir, test_dir, transform, batch_size=8):
    transformTrain = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.uint8, scale=True),
        v2.Resize((360, 360)),
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

    transformTest = v2.Compose([
        v2.ToImage(), 
        v2.ToDtype(torch.float32, scale=True), 
        v2.Resize((360, 360))
    ])
    
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