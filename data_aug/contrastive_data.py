#%% Imports
import os
import tarfile
from torchvision.transforms import transforms
from torchvision.datasets import ImageFolder
from torchvision import transforms
from data_aug.key_and_query import KeyAndQuery
from data_aug.sobel_filter import SobelFilterKornia

#%% Class
class DataAugContrastive:
    """ Tools for extracting, organising and augmenting data ready 
    to assemble as DataLoader for constrastive learning.
    
    
    Attributes
    ----------------
    
        root_folder: str
            Folder directory of a single .tar or .tar.gz data files


    Methods
    ----------------

        unpack_tar_data(self, data_path, data_name):
            Unpack .tar or .tar.gz files into same root folder as data_path

        eval_transforms(size, s):
            Basic transforms to ready images for passing into trained encoder 

        simclr_transforms(size, s):
            Final transforms chosen for constrastive learning

        get_data(self, path, name, no_view=None):
            Take images from specified file and complie into ImageFolder



    """

    def __init__(self, root_folder):
        self.root_folder = root_folder

    def unpack_tar_data(self, data_path, data_name):
        """Unpack .tar or .tar.gz files into same root folder as data_path.
        """

        filename = f'{data_path}\\{data_name}' #extract directory

        # create data folder in 'data_path' directory
        if filename.endswith("tar.gz"):
            tar = tarfile.open(filename, "r:gz")
            tar.extractall(data_path)
            tar.close()
        elif filename.endswith("tar"):
            tar = tarfile.open(filename, "r:")
            tar.extractall(data_path)
            tar.close()


    @staticmethod
    def eval_transforms(size, s=1):
        """Basic transforms to ready images for passing into trained encoder.
        """
        
        data_transforms = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])])

        return data_transforms

    @staticmethod
    def simclr_transforms(size, s=1):
        """Example transform package for constrastive learning, 
        some determined by SimCLR paper (https://arxiv.org/abs/2002.05709), others by testing.

        Some transform options left as comments for testing. Normalized val(mean,std) taken from ImageNet given pretrained resnet18.

        CAUTION: order of transforms matters.
        """

        # stdDev range of blur encourages model to ignore fine-grained texture and focus on higher-level structure, 
        sobel = SobelFilterKornia()
        blur = transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))
        color_alter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        data_transforms = transforms.Compose([
            # transforms.RandomResizedCrop(size=size),
            transforms.RandomResizedCrop(size=size, scale=(0.8,1.0)),
            # transforms.CenterCrop(size=size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([color_alter], p=0.8),
            transforms.RandomApply([blur], p=0.5),
            transforms.ToTensor(),
            transforms.RandomApply([sobel], p=0.5),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        return data_transforms
    
    def get_data(self, path, name, size, no_view=None):
        """Take images from specified file and complie into ImageFolder.
        """
               
        # unpack provided path and organise into usable data set
        if len(os.listdir(path)) == 1:
            self.unpack_tar_data(path, name)

        # remove extension and assemble final directory
        if name.endswith(".tar.gz"):
            base = name[:-7]
        elif name.endswith(".tar"):
            base = name[:-4]
        else:
            base = os.path.splitext(name)[0]
        
        name = f'{base}/data'

        # organise into ImageFolder assuming data structure is: name/<class-i>/images...
        if no_view != None:
            dataset = ImageFolder(root= f'{path}\\{name}', 
                                transform = KeyAndQuery(
                                    self.simclr_transforms(size),
                                    no_view))
        else:
            dataset = ImageFolder(root= f'{path}\\{name}', 
                                transform = KeyAndQuery(
                                    self.eval_transforms(size),
                                    1))

        return dataset