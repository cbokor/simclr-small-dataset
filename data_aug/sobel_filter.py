#%% Imports

import kornia
import kornia.filters as KF
import torch

#%% Class

class SobelFilterKornia():
    """ Sobel filter implemented with Kornia, structured to be compatible with torchvision transform pipelines.
    
    Methods
    ----------------

        __call__(self, image):
            Apply sobel filter to provided image tesnor
    
    """
    
    def __call__(self, image):
        """ Apply sobel filter to provided image tesnor
        
        Parameters
        ----------------

            image : torch.Tensor
                A 3D tensor of shape [C, H, W] representing an image. Expected to be in RGB format.
        """

        image = image.unsqueeze(0)  # add batch dimension
        gray = kornia.color.rgb_to_grayscale(image) # convernt to grayscale
        edges = KF.sobel(gray)  # apply sobel edge detection, returning graident vectors
        
        if edges.shape[1] == 1:
            edges_mag = edges   # pass if already single channel
        else:
            edges_mag = torch.sqrt(edges[:,0] ** 2 + edges[:,1] ** 2) # extract magnitudes of gradients

        # optional argument left below depending on transform formating/order (currently, projector head normalizes)
        # =========================================

        # edges_mag = transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                         std=[0.229, 0.224, 0.225])(edges_mag)

        # =========================================

        #normalize to [0,1], 1e-5 to avoid devision by zero
        edges_mag = (edges_mag - edges_mag.min()) / (edges_mag.max() - edges_mag.min() + 1e-5)

        edges_mag = edges_mag.repeat(1, 3, 1, 1) #match rgb format, repeating grey scale accross 3 channels

        return edges_mag.squeeze(0)  # remove batch dim