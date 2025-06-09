#%% Imports

import numpy as np

#%% Initialization

np.random.seed(0)

#%% Class

class KeyAndQuery(object):
    """Generate multiple random augmented views (e.g., key and query) of a single image using a specified transform pipeline.

    Attributes
    ----------------
    
        transform_pack : callable
            Torchvision-style transform pipeline applied to input image.

        no_view : int
            Number of augmented views to generate from a single image (default:2)


    Methods
    ----------------

        __call__(x):
            Applies transform pipeline to input image `no_view` times, returning list of augmented views.
            
    """

    def __init__(self, transform_pack, no_view=2):
        self.transform_pack = transform_pack
        self.no_view = no_view

    def __call__(self, x):
        return[self.transform_pack(x) for i in range(self.no_view)]