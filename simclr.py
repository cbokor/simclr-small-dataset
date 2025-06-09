#%% Imports 

import torch
import utils
from datetime import datetime
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter



#%% Class

class SimCLR():
    """Wrapper class for managing SimCLR training pipeline components including model, optimizer, 
    scheduler, loss function, and TensorBoard logging.
    
    Attributes
    ----------------
    
        args : Namespace or dict
            argparse container including training configuration.

        encoder : nn.Module
            Neural network encoder model.

        optimizer : torch.optim.Optimizer
            Optimizer for updating model parameters during training.

        scheduler : torch.optim.lr_scheduler._LRScheduler
            Learning rate scheduler for management during training.

        criterion : nn.Module
            Loss function used as the training criterion.

        writer : SummaryWriter
            TensorBoard SummaryWriter instance for logging training metrics/hyperparams.


    Methods
    ----------------

        extract_checkpoint(self):
            Load previous specified sim_clr checkpoint into encoder.

        nce_loss(self, features, actual_batch_size, check):
            Formulate normalized temperature-scaled cross entropy loss (NCE).

        train(self, train_loader):
            Main traning loop for simCLR, optional AMP.


    """
    
    def __init__(self, *args, **kwargs):
        """Instance input properties and initiate SummaryWriter
        """

        # Unpack args (list/tuple) and kwargs (dict) containers respectively 
        self.args = kwargs.get('args',None)

        self.encoder = kwargs['model'].to(self.args.device)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)

        # Summary writer initialize and checkpoint info
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        exp_name = f"simclr_lr_{self.args.lr}_bs_{self.args.batch_size}_{timestamp}"
        self.writer = SummaryWriter(log_dir=f"runs/{exp_name}")

    def extract_checkpoint(self):
        """ load previous specified sim_clr checkpoint into encoder, default: args.checkpoint_path
        """

        print(f"Loading checkpoint from {self.args.checkpoint_path}")
        checkpoint = torch.load(self.args.checkpoint_path)

        self.encoder.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.args.start_epoch = checkpoint['epoch'] + 1

        # Optional: Load scheduler & best loss
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if 'best_loss' in checkpoint:
            self.args.best_loss = checkpoint['best_loss']
    
        print(f"Resumed from epoch {self.args.start_epoch}")


    def nce_loss(self, features, actual_batch_size, check):
        """ Formulate normalized temperature-scaled cross entropy loss (NCE)
        as per eq-1 in paper 'https://arxiv.org/abs/2002.05709'"""
        
        # Assign same label to all augmented views of the original sample. labels.shape == [actual_batch_size * n_views] = [N]
        if (not self.args.drop_last) and check and (actual_batch_size != self.args.batch_size):
            labels = torch.cat([torch.arange(actual_batch_size) for i in range(self.args.no_view)], dim=0)
        else:
            labels = torch.cat([torch.arange(self.args.batch_size) for i in range(self.args.no_view)], dim=0)

        # Assemble positive pair mask (pairwise label comparisons) for constrastive learning of labels. 
        # input: tensor of class labels; output: booleon mask converted to float tensor.
        # 1s = positive pairs; 0's = negative pairs. labels.shape == [N, N]
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()

        labels = labels.to(self.args.device)    # move labels to chosen device

        # convert each feature into a unit vector via default L2 norm.
        # (stabilizes training; consine_sim~dotProduct; prevents magnitude affecting similarity, only direction) 
        # features.shape == [N, feature_dim]
        features = torch.nn.functional.normalize(features, dim=1)

        # construct simularity matrix. similarity_matrix.shape == [N, N]
        # element (i,j) = cosine similarity of feature vec i & feature vec j 
        similarity_matrix = torch.matmul(features, features.T)

        # Two options for size based on skipping final batch.
        if not self.args.drop_last:
            N = features.shape[0]   # drop_last=False
        else:
            N = self.args.batch_size * self.args.no_view    # drop_last=True

        # discard main diagonal (self-simularities) from both labels and sim matrix
        # assert simularity_matrix.shape == labels.shape, i.e., [N, N-1]. 
        temp_mask = torch.eye(N, dtype=torch.bool).to(self.args.device)
        similarity_matrix = similarity_matrix[~temp_mask].view(N, -1)
        labels = labels[~temp_mask].view(N, -1)

        # boolean mask of positives. pos.shape == [N, num_positives_per_sample] 
        pos = similarity_matrix[labels.bool()].view(labels.shape[0], -1)    

        # boolean mask of negatives. neg.shape == [N, N - num_positives_per_sample - 1]
        neg = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1) 

        # stitch together into a logits matirx (each row = one training example)
        logits = torch.cat([pos, neg], dim=1)

        # Create target labels for CrossEntropyLoss, postiive simularity = 0
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.args.device)

        # Sharpen/smooth peaks for softmax
        logits = logits / self.args.temperature

        return labels, logits 


    def train(self, train_loader):
        """ main traning loop for simCLR, optional AutoMixedPrecision (AMP) included"""

        # Class to handle dynamic loss scaling
        scaler = GradScaler(enabled=self.args.amp)                                      

        n_iter = 0

        # repeatedly cycle over data 'args.epochs-self.args.start_epoch' times
        for epoch_counter in range(self.args.start_epoch, self.args.epochs):
            
            print(f'Epoch Counter: {epoch_counter}')

            for images, labels in tqdm(train_loader):   # iterate through all batches in train_loader, display 'tqdm' progress bar
                
                images = torch.cat(images, dim=0)       # concatenate list (i.e., base batch and augmented views) of images
                images = images.to(self.args.device)    # move images tensor to chosen device (must be same as model location)

                # Variables to identify final batch when args.drop_last=False
                actual_batch_size = images.shape[0] // self.args.no_view
                check = images.shape[0] != self.args.batch_size * self.args.no_view

                # Context manager: enable/disable AMP, i.e., dynamicly apply float16 precision or normal float32 
                with autocast(enabled=self.args.amp):
                    features = self.encoder(images)
                    labels, logits = self.nce_loss(features, actual_batch_size, check)
                    loss = self.criterion(logits, labels)
                
                self.optimizer.zero_grad()      # zero out gradients to prevent accumulaion (PyTorch default is to accumulate)
                scaler.scale(loss).backward()   # compute grads and scale to prevent vanishing grads/underflow in float16 (AMP)
                scaler.step(self.optimizer)     # if AMP: unscale grads, then update encoder params
                scaler.update()                 # Check if grads valid, adjust scaler dynamiclly (noIssues=increase, NaNs/Inf=decrease)

                # log every n iterations specified by args.log_interval
                if n_iter % self.args.log_interval == 0:
                    self.writer.add_scalar('loss', loss, global_step=n_iter)
                    self.writer.add_scalar('learning_rate', self.scheduler.get_last_lr()[0], global_step=n_iter)
                    self.writer.flush

                # store the best model identified in run
                if loss < self.args.best_loss:

                    self.args.best_loss = loss

                    utils.save_checkpoint({
                        'arch': self.args.encoder,
                        'epoch': epoch_counter,
                        'model_state_dict': self.encoder.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'best_loss': self.args.best_loss,
                        'scheduler_state_dict': self.scheduler.state_dict(),
                    }, f'checkpoints/best_model.pth.tar')

                n_iter += 1

            # warmup period (scheduler holds learning rate constant for first 'args.warmup_epochs' epochs)
            if (epoch_counter) >= self.args.warmup_epochs:
                self.scheduler.step()

            # regular checkpoint storage per 10 epochs
            if epoch_counter % 10 == 0:
                utils.save_checkpoint({
                    'arch': self.args.encoder,
                    'epoch': epoch_counter,
                    'model_state_dict': self.encoder.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                }, f'checkpoints/checkpoint_epoch_{epoch_counter}.pth.tar')

        self.writer.close()