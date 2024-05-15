import torch 
import torch.nn as nn
import numpy as np
import pandas as pd 

from sklearn import metrics
from sklearn.model_selection import train_test_split

import timm
import albumentations

from torchret import Model 

configs = {
    'epochs' : 10,
    'lr' : 1e-4,
    'eta_min' : 1e-6,
    'T_0' : 10,
    'epochs' : 10,
    'step_scheduler_after' : 'epoch',

    'train_bs' : 64,
    'valid_bs' : 64,

    'num_workers' : 0,
    'pin_memory' : False,

    'model_name' : 'resnet10t',
    'pretrained' : True,
    'num_classes' : 10,
    'in_channels' : 1,
    'device' : 'mps',

    'model_path' : 'digit-recognizer.pt',
    'save_best_model' : 'on_eval_metric',
    'save_on_metric' : 'accuracy',
    'save_model_at_every_epoch' : False,

}

class DigitRecognizerDataset:
    def __init__(self, df, augmentations):
        self.df = df
        self.targets = df.label.values
        self.df = self.df.drop(columns=["label"])
        self.augmentations = augmentations

        self.images = self.df.to_numpy(dtype=np.float32).reshape((-1, 28, 28))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        targets = self.targets[item]
        image = self.images[item]
        image = np.expand_dims(image, axis=0)

        return {
            "images": torch.tensor(image, dtype=torch.float),
            "targets": torch.tensor(targets, dtype=torch.long),
        }

class DigitRecognizerModel(Model):
    def __init__(self):
        super().__init__()

        self.model = timm.create_model(
            model_name = configs['model_name'],
            pretrained=configs['pretrained'],
            in_chans=configs['in_channels'],
            num_classes=configs['num_classes'],
        )

        self.num_workers = configs['num_workers']
        self.pin_memory = configs['pin_memory']
        self.step_scheduler_after = configs['step_scheduler_after']

        self.model_path = configs['model_path']
        self.save_best_model = configs['save_best_model']
        self.save_on_metric = configs['save_on_metric']
        self.save_model_at_every_epoch = configs['save_model_at_every_epoch']

    def monitor_metrics(self, outputs, targets):
        device = targets.device.type
        outputs = np.argmax(outputs.cpu().detach().numpy(), axis=1)
        targets = targets.cpu().detach().numpy()
        acc = metrics.accuracy_score(targets, outputs)
        acc = torch.tensor(acc, device=device)
        return {"accuracy": acc}
    
    def monitor_loss(self, outputs, targets):
        loss = nn.CrossEntropyLoss()(outputs, targets)
        return loss

    def fetch_optimizer(self):
        opt = torch.optim.SGD(
            self.parameters(),
            lr=configs['lr'],
            momentum=0.9,
        )
        sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt, 
            T_0 = configs['T_0'],
            eta_min = configs['eta_min'],
            T_mult= 1
        )
        return opt, sch

    def forward(self, images, targets=None):
        x = self.model(images)
        if targets is not None:
            loss = self.monitor_loss(x, targets)
            metrics = self.monitor_metrics(x, targets)
            return x, loss, metrics
        return x, 0, {}
    
df = pd.read_csv('train.csv')
train, test = train_test_split(df, test_size=0.2)

train_augs = albumentations.Compose(
        [
            albumentations.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                p=1.0,
            ),
        ],
        p=1.0,
    )

valid_augs = albumentations.Compose(
        [
            albumentations.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                p=1.0,
            ),
        ],
        p=1.0,
    )

train_dataset = DigitRecognizerDataset(df = train, augmentations = train_augs)
valid_dataset = DigitRecognizerDataset(df = test, augmentations = valid_augs)

model = DigitRecognizerModel()
model.fit(train_dataset, valid_dataset, device = configs['device'], epochs = configs['epochs'])

model.load('digit-recognizer.pt', device = 'cpu')
predictions = model.predict(valid_dataset)

final_preds = []
for preds in predictions:
    final_preds.append(preds)
final_preds = np.concatenate(final_preds)