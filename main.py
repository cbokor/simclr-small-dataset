#%% Tensorboard setep
# (1) tensorboard --logdir="...<ThisRepoFolder>\runs"
# (2) http://localhost:6006

#%% Imports

import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import utils
import samplers
import torch.nn as nn

from torch.utils.data import DataLoader
from data_aug.contrastive_data import DataAugContrastive
from torchvision import models
from encoders.resnet_simclr import ResNetSimCLR
from simclr import SimCLR

#%% Initialize

# extract all avilable encoder models
encoder_names = sorted( name for name in models.__dict__
                       if name.islower() and not name.startswith("__")
                       and callable(models.__dict__[name]))             

# construct parser:
parser = argparse.ArgumentParser(description = 'SimCLR via PyTorch')
parser.add_argument('-data', metavar='DIR', default = f'{os.getcwd()}\data',
                    help='path to dataset')
parser.add_argument('-dataset-name', default='example_data_folder.tar',
                    help='dataset name', choices = os.listdir(f'{os.getcwd()}\data'))
parser.add_argument('--workers', default=4, type=int,
                    help='number of workers for dataloader (default: 4)')
parser.add_argument('--no-view', default=2, type=int, metavar='N',           
                    help='Number of views for contrastive learning training.')
parser.add_argument('--gpu-index', default=0, type=int,
                    help='Use Gpu-index(0) or not if (-1)/None (default: 0).')
parser.add_argument('-b', '--batch-size', default=256, type=int, metavar='N',
                    help='size of image batches per epoch')
parser.add_argument('-e', '--encoder', metavar='ENC', default='resnet18',
                    choices = encoder_names,
                    help='model architecture: ' + ' | '.join(encoder_names) +
                         ' (default: resnet18)')
parser.add_argument('-wd', '--weight-decay', default=1e-5, type=float,
                    metavar='W', help='weight decay (default: 1e-5)',
                    dest='weight_decay')
parser.add_argument('-lr', '--learning-rate', default=1e-3, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='total number of training passes over full dataset (default: 200)')
parser.add_argument('--warmup-epochs', default=10, type=int,
                    help='initial set of warmup epochs (i.e., learning rate held constant)')
parser.add_argument('--temperature', default=0.05, type=float,
                    help='softmax temperature (default: 0.05)')
parser.add_argument('--drop-last', default=False, type=bool,
                    help='drop final batch of training data (default: False)')
parser.add_argument('-amp', default = False, type=bool,
                    help='Inlucde Auto Mixed Precision training (default: False)')
parser.add_argument('--log-interval', default=1, type=int,
                    help='log interval every n training steps')
parser.add_argument('--preload', default=False, type=bool,
                    help='preload previously trained model (default: False)')
parser.add_argument('--checkpoint_path', default=rf'{os.getcwd()}\checkpoints\best_model.pth.tar',
                    help='path for loading chosen previous checkpoint (default: best_model.pth.tar)')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='epoch counter starting value for pre-loading (default: 0)')
parser.add_argument('--best-loss', default=float('inf'), type=float,
                    help='best loss identified so far (default: inf)')
parser.add_argument('--image-size', default=224, type=int,
                    help='output image size once processed by transforms for dataloader (default: 224)')
parser.add_argument('--pretrain', default=True, type=bool,
                    help='pretrain simclr off own data before loading from checkpoint')


#%% Methods

def main():
    """Main operating script.
    """

    args = parser.parse_args()

    # check if gpu training is available
    if torch.cuda.is_available():
        args.device = torch.device('cuda')
        num_workers = min(args.workers, os.cpu_count())
        cudnn.deterministic = True
        cudnn.benchmark = True
    else:
        args.device = torch.device('cpu')
        num_workers = 0 # keep safe for CPU-only, avoid multiprocessing issues
        args.gpu_index = -1
        
    print("Device:", args.device)
    print("Number of workers assigned:", num_workers)
    print("Total number of epochs:", args.epochs)
    print("Number of warmup epochs:", args.warmup_epochs)

    # Organise data into ImageLoader
    data = DataAugContrastive(args)
    train_set = data.get_data(args.data, args.dataset_name, args.image_size, args.no_view)

    # Initialize weighted sampling to mitigate uneven data groups
    sampler = samplers.weighted_sampling(train_set)

    # Organise data into DataLoader: 
    # given feature map goal and tiny data set, all data is assigned for training (no val)

    #train_set, test_set = torch.utils.data.random_split(train_set, [**, **])
    train_loader = DataLoader(train_set, batch_size = args.batch_size,
                              drop_last=args.drop_last,
                              num_workers = num_workers,
                              #shuffle=True,
                              sampler=sampler,
                                )

    # Optional batch view: 
    # utils.view_example_batch(train_loader, args.batch_size)

    # Assign encoder from options
    encoder = ResNetSimCLR(base_encoder = args.encoder)

    # Assign optimizer: default is 'Adam' with weight decay due to tiny dataset (Stable, fast, regularised for proof of concept)
    optimizer = torch.optim.Adam(encoder.parameters(), args.lr, weight_decay = args.weight_decay)

    # Scheduler to dictate the learning rate with CosineAnnealingLR (i.e., learning rate annealed accros args.epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = args.epochs, eta_min = 1e-6,
                                                            last_epoch = -1)

    # Context manager to temporarily set active CUDA device. no-op if gpu_index = negative(int) or None.
    with torch.cuda.device(args.gpu_index):

        # Assemble and train full pipeline as per simclr_v1.0 (https://arxiv.org/abs/2002.05709)
        simclr = SimCLR(model = encoder, optimizer = optimizer, scheduler = scheduler, args = args)

        # Pre-train Simclr yes/no
        if args.pretrain==True:
            
            # Load prior checkpoint if exists & requested
            if os.path.exists(args.checkpoint_path) and args.preload==True:
                simclr.extract_checkpoint()

            simclr.train(train_loader)

    
    # Load specified pretrained model (default: best)
    simclr.extract_checkpoint()

    # Extract encoder, removing dim=128 projection head to expose generalisable semantic features (output dim=512)
    encoder = simclr.encoder
    encoder.backbone.fc = nn.Identity()
    encoder.eval()

    # Create evaluation data loader of all provided images
    eval_set = data.get_data(args.data, args.dataset_name, args.image_size)
    eval_loader = DataLoader(eval_set, batch_size = args.batch_size,
                             drop_last=args.drop_last,
                             num_workers = num_workers,
                             shuffle=False
                             )
    
    # Optional batch view:
    # utils.view_example_batch(eval_loader, args.batch_size)

    # Extract features and plot feature map for linear seperation (pca) and nonlinear structure/clustering (t-SNE)
    features, labels = utils.extract_features(eval_loader, encoder, args.device)
    utils.plot_pca(features, labels)
    utils.plot_tsne(features, labels)




#%% Script
if __name__ == "__main__":  # only operate script if called directly (i.e., not as module, __name__ = "my_module")
    
    main()