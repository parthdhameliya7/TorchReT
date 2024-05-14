import torch.nn as nn 
import torch 
from torchret.utils import AverageMeter

from tqdm import tqdm
import numpy as np
import random

class Model(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        """
        Constructor for the Model class.

        Initializes various attributes used during training and evaluation.

        Attributes:
            optimizer (Optimizer or None): Holds the optimizer instance.
            scheduler (LRScheduler or None): Holds the scheduler instance.
            trainloader (DataLoader or None): DataLoader for training dataset.
            validloader (DataLoader or None): DataLoader for validation dataset.
            num_workers (int or None): Number of workers for data loading.
            pin_memory (bool or None): Whether to pin memory for DataLoader.
            collate_fn (function or None): Collate function for DataLoader.
            step_scheduler_after (str): Step scheduler after each batch or epoch.
            current_epoch (int): Current epoch during training.
            current_train_step (int): Current training step.
            current_valid_step (int): Current validation step.
            device (str or None): Device for model and data.
            fp16 (bool or None): Whether to use mixed precision training.
            scaler (torch.cuda.amp.GradScaler or None): Gradient scaler for mixed precision training.
            mixup (float): Mixup augmentation probability.
            cutmix (float): Cutmix augmentation probability.
            sam_training (bool or None): Whether to use Sharpness-Aware Minimization.
            swa_training (bool or None): Whether to use Stochastic Weight Averaging.
            monitor_metric_at_end (str or None): Metric to monitor at the end of each epoch.
            save_model_at_every_epoch (bool or None): Whether to save model at every epoch.
            model_path (str or None): Path to save the model.
            weights_only (bool or None): Whether to save only the model weights.
            save_best_model (str or None) : Whether to save model based on (on_eval_loss or on_eval_metric)
            save_on_metric (str or None) : Model saving on which metric
            metric_path (str or None) : additional metric path to differ from original model path
        """
        super().__init__(*args, **kwargs)

        self.optimizer = None
        self.scheduler = None 
        self.trainloader = None
        self.validloader = None
        self.num_workers = None
        self.pin_memory = None 
        self.collate_fn = None 
        self.step_scheduler_after = 'batch'
        self.current_epoch = 0 
        self.current_train_step = 0 
        self.current_valid_step = 0 
        self.device = None 
        self.fp16 = None 
        self.scaler = None
        self.mixup = 1.0
        self.cutmix = 1.0
        self.sam_training = None 
        self.swa_training = None 
        self.monitor_metric_at_end = None
        self.save_model_at_every_epoch = None
        self.save_best_model = None
        self.save_on_metric = None
        self.model_path = None
        self.weights_only = None
        self.metric_path = None

    def forward(self, *args, **kwargs):
        """
        Forward pass of the model.
        """
        super().forward(*args, **kwargs) 

    def monitor_metrics(self, *args, **kwargs):
        """
        Method to monitor additional metrics during training or evaluation.
        """
        return 
    
    def fetch_optimizer(self, *args, **kwargs):
        """
        Method to fetch the optimizer for training.
        """
        return 
    
    def fetch_scheduler(self, *args, **kwargs):
        """
        Method to fetch the scheduler for adjusting learning rate during training.
        """
        return 
    
    def get_mixup(self, *args, **kwargs):
        """
        Method to apply mixup augmentation to the data.
        """
        return 
    
    def get_cutmix(self, *args, **kwargs):
        """
        Method to apply cutmix augmentation to the data.
        """
        return 
    
    def model_fn(self, data):
        """
        Forward pass of the model with data processing.

        Args:
            data (dict): Dictionary containing input data tensors.

        Returns:
            output: Model output tensor.
            loss: Loss tensor.
            metrics: Dictionary containing additional metrics.
        """
        for k, v in data.items():
            data[k] = v.to(self.device)
        if self.fp16 is not None:
            with torch.cuda.amp.autocast():
                output, loss, metrics = self(**data)
        else:
            output, loss, metrics = self(**data)
        return output, loss, metrics

    def train_one_step(self, data):
        """
        Perform one step of training.

        Args:
            data (dict): Dictionary containing input data tensors.

        Returns:
            loss: Loss tensor.
            metrics: Dictionary containing additional metrics.
        """
        # Apply mixup or cutmix augmentation
        if random.uniform(0, 1) > self.mixup:
            data = self.get_mixup(data)
        elif random.uniform(0, 1) > self.cutmix:
            data = self.get_cutmix(data)

        # Apply train step with fp16 precision training 
        if self.fp16 is not None:
            self.optimizer.zero_grad()
            _, loss, metrics = self.model_fn(data)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

        # Apply train step with Sharpness-Aware Minimization 
        elif self.sam_training is not None:
            _, loss1, metrics = self.model_fn(data)
            loss1.backward()
            self.optimizer.first_step(zero_grad = True)

            _, loss2, metrics = self.model_fn(data)
            loss2.backward()
            self.optimizer.second_step(zero_grad = True)

        else:
            self.optimizer.zero_grad()
            _, loss, metrics = self.model_fn(data)
            loss.backward()
            self.optimizer.step()

        if self.scheduler is not None:
            if self.step_scheduler_after == 'batch':
                self.scheduler.step()
                
        return loss, metrics
    
    def valid_one_step(self, data):
        _, loss, metrics = self.model_fn(data)
        return loss, metrics

    def train_one_epoch(self, dataloader):
        self.train()
        losses = AverageMeter()
        tracker = tqdm(dataloader, total = len(dataloader))
        for batch_id, data in enumerate(tracker):
            loss, metrics = self.train_one_step(data)
            losses.update(loss.item(), dataloader.batch_size)
            if batch_id == 0:
                metrics_meter = {k : AverageMeter() for k in metrics}
            monitor = {}
            for m_m in metrics_meter:
                metrics_meter[m_m].update(metrics[m_m].item(), dataloader.batch_size)
                monitor[m_m] = metrics_meter[m_m].avg
            tracker.set_postfix(epoch = self.current_epoch, loss='%.6f' %float(losses.avg), stage="train", **monitor)
            self.current_train_step += 1

        if self.swa_training is not None:
            if self.current_epoch + 1 >= self.swa_start:
                self.swa_model.update_parameters(self)
            else:
                if self.scheduler is not None:
                    if self.step_scheduler_after == 'epoch':
                        self.scheduler.step()
        else:
            if self.scheduler is not None:
                if self.step_scheduler_after == 'epoch':
                    self.scheduler.step()
        tracker.close()
        return losses.avg, monitor
    
    def valid_one_epoch(self, dataloader):
        self.eval()
        losses = AverageMeter()
        tracker = tqdm(dataloader, total = len(dataloader))
        with torch.no_grad():
            for batch_id, data in enumerate(tracker):
                loss, metrics = self.valid_one_step(data)
                losses.update(loss.item(), dataloader.batch_size)
                if batch_id == 0:
                    metrics_meter = {k : AverageMeter() for k in metrics}
                monitor = {}
                for m_m in metrics_meter:
                    metrics_meter[m_m].update(metrics[m_m].item(), dataloader.batch_size)
                    monitor[m_m] = metrics_meter[m_m].avg
                tracker.set_postfix(epoch = self.current_epoch, loss='%.6f' %float(losses.avg), stage="eval", **monitor)
                self.current_valid_step += 1
            tracker.close()
            return losses.avg, monitor
        
    def predict_one_step(self, data):
        output, _, _ = self.model_fn(data)
        return output
        
    def process_output(self, output):
        output = output.cpu().detach().numpy()
        return output
        
    def predict(self, dataset, batch_size = 16, collate_fn = None):
        if next(self.parameters()).device != self.device:
            self.to(self.device)

        if self.num_workers is None:
            n_jobs = 0

        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, num_workers=n_jobs
        )

        if self.training:
            self.eval()

        tracker = tqdm(data_loader, total=len(data_loader))

        for _, data in enumerate(tracker):
            with torch.no_grad():
                out = self.predict_one_step(data)
                out = self.process_output(out)
                yield out 
                tracker.set_postfix(stage = 'test')
        tracker.close()
        
    def save(self, model_path, weights_only = False):
        model_state_dict = self.state_dict()
        if weights_only:
            torch.save(model_state_dict, model_path)
            return 
        if self.optimizer is not None:
            opt_state_dict = self.optimizer.state_dict()
        else:
            opt_state_dict = None 
        if self.scheduler is not None:
            sch_state_dict = self.scheduler.state_dict() 
        else:
            sch_state_dict = None 
        model_dict = {}
        model_dict['state_dict'] = model_state_dict 
        model_dict['optimizer'] = opt_state_dict
        model_dict['scheduler'] = sch_state_dict
        model_dict['epoch'] = self.current_epoch 
        model_dict['fp16'] = self.fp16
        model_dict['sam_training'] = self.sam_training
        model_dict['swa_training'] = self.swa_training
        model_dict['train_metrics'] = self.train_metrics
        model_dict['train_loss'] = self.train_loss
        if self.validloader is not None:
            model_dict['valid_loss'] = self.valid_loss
        elif self.valid_metrics is not None:
            model_dict['valid_metrics'] = self.valid_metrics

        torch.save(model_dict, model_path)
        print(f'Model Saved at {model_path}')

    def load(self, model_path, weights_only = False, device = 'cuda'):
        self.device = device 
        if next(self.parameters()).device != self.device:
            self.to(self.device)
        model_dict = torch.load(model_path, map_location=torch.device(device))
        if weights_only:
            self.load_state_dict(model_dict)
            print('Model Loaded succesfully!')
        else:
            self.load_state_dict(model_dict['state_dict'])
            print('Model Loaded succesfully!')

    def fit(
            self,
            train_dataset,
            valid_dataset = None,
            device = 'cuda',
            epochs = 10,
            train_bs = 64,
            valid_bs = 64,
    ):

        if self.trainloader is None:
            self.trainloader = torch.utils.data.DataLoader(
                train_dataset,
                train_bs,
                shuffle = True
            )
        
        if self.validloader is None:
            self.validloader = torch.utils.data.DataLoader(
                valid_dataset,
                valid_bs,
                shuffle = True
            )
        if next(self.parameters()).device != self.device:
            self.to(self.device)
        
        if self.fp16:
            self.scaler = torch.cuda.amp.GradScaler()

        self.optimizer = self.fetch_optimizer()
        self.scheduler = self.fetch_scheduler()
        self.current_epoch += 1
        self.best_loss = np.Inf
        self.best_score = 0.0


        for _ in range(epochs):
            self.train_loss, self.train_metrics = self.train_one_epoch(self.trainloader)
            if self.validloader is not None:
                self.valid_loss, self.valid_metrics = self.valid_one_epoch(self.validloader)
            self.current_epoch += 1

            if self.save_best_model is not None:
                if self.save_best_model == 'on_eval_loss':
                    if self.valid_loss < self.best_loss:
                        self.save(f'{self.metric_path}_{self.model_path}', self.weights_only)
                        print(f'Model was saved based {self.save_best_model} with {self.valid_loss} loss')
                elif self.save_best_model == 'on_eval_metric':
                    if self.save_on_metric in self.train_metrics:
                        if self.valid_metrics[self.save_on_metric] > self.best_score:
                            self.save(f'{self.metric_path}_{self.model_path}', self.weights_only)
                            print(f'Model was saved based {self.save_best_model} with {self.valid_metrics[self.save_on_metric]} {self.save_on_metric}')
                
            if self.save_model_at_every_epoch is not None:
                self.save(self.model_path, self.weights_only)




