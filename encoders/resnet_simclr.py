#%% Imports
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights


#%% Class

class ResNetSimCLR(nn.Module):

    """
    Class object for chosen encoder model framed as a child class of torch.nn.Module.
    Implements SimCLR-style architecture using ResNet backbone and a projector head.

    Attributes
    ----------
        resnet_dict : dict
            Dictionary of supported ResNet models with torchvision weights.
    
        backbone : nn.Module
            Base encoder model with final classification layer replaced by an MLP projection head.

    Methods
    -------
        get_base_encoder(self, model_name):
            Assigns and returns the chosen ResNet encoder from the available dictionary.

        forward(self, x):
            Passes input through the encoder backbone (including projection head by default).
    """

    def __init__(self, base_encoder):
        """Initializes ResNetSimCLR model with a ResNet backbone and custom projection head.
        """

        super(ResNetSimCLR, self).__init__()    # call constructur of parent class nn.module

        # dict of currently supported/tested models (assumes pretrained)
        self.resnet_dict = {"resnet18": models.resnet18(weights=ResNet18_Weights.DEFAULT),
                            }

        self.backbone = self.get_base_encoder(base_encoder) # assign backbone
        enc_dim = self.backbone.fc.in_features # get encoder output dimension

        # add mlp projection head
        # note: dont want to hyjack resnets fc layer as it was designed for classification,
        # instead entire head replaced with 128 proj head as per sml-clr paper for constrastive loss.
        projection_dim = 128
        self.backbone.fc = nn.Sequential(
            nn.Linear(enc_dim, enc_dim),
            nn.ReLU(),
            nn.Linear(enc_dim, projection_dim)
        )



    def get_base_encoder(self, model_name):
        """Assigns and returns the chosen ResNet encoder from the available dictionary.
        """

        model = self.resnet_dict[model_name]

        return model
    
    def forward(self, x):
        """Passes input through the encoder backbone (including projection head by default).
        """
        return self.backbone(x)