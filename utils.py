#%% Imports
import torch
import matplotlib.pyplot as plt
import torchvision
import seaborn
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

#%% Methods

def plot_pca(features, labels):
    """Plot 2d principle component analysis (pca) of evaluated
    encoder features using class labels to indicate linear seperation.
    """

    scaler = StandardScaler()
    features = scaler.fit_transform(features.numpy())

    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(features)

    plt.figure(figsize=(10, 8))
    seaborn.scatterplot(x=pca_result[:,0], y=pca_result[:,1], hue=labels.numpy(), palette='deep')
    plt.legend()
    plt.title("PCA Plot")

    return plt.show()


def plot_tsne(features, labels, epoch=None):
    """Plot 2d t-distributed Stochastic Neighbor Embedding (t-SNE) of evaluated
    encoder features using class labels to indicate nonlinear structure/clusters.
    """

    features = F.normalize(features, p=2, dim=1)

    # tsne = TSNE(n_components=2 , perplexity = 30, n_iter=2000, random_state=42)
    tsne = TSNE(n_components=2 , perplexity = 10, init='pca', n_iter=5000, random_state=42)
    tsne_result = tsne.fit_transform(features.numpy())

    plt.figure(figsize=(10,8))
    seaborn.scatterplot(x=tsne_result[:,0],
                        y=tsne_result[:,1],
                        hue=labels.numpy(),
                        palette='deep',
                        s=60)
    
    if epoch is not None:
        plt.title(f't-SNE at epoch {epoch}')
    else:
        plt.title(f't-SNE Plot')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.legend()
    plt.tight_layout()

    return plt.show()

def extract_features(dataloader, encoder, device):
    """Extract featues from provided encoder entering specified dataloader of images.
    """
    features = []
    labels = []

    encoder.to(device)
    encoder.eval() #ensure its in inferance mode

    with torch.no_grad():
        for images, target in tqdm(dataloader):
            if isinstance(images, list):
                images = torch.stack(images)
            images = images.view(-1, *images.shape[2:])
            images = images.to(device)
            outputs = encoder(images)
            features.append(outputs.cpu())
            labels.append(target)

    features = torch.cat(features, dim=0)
    labels = torch.cat(labels, dim=0)

    return features, labels

def view_example_batch(data_loader, batch_size):
    """Preview a batch of images from provided dataloader.
    """

    # Extract one batch
    images, _ = next(iter(data_loader))
    
    # make a grid of images for visualization
    img_grid = torchvision.utils.make_grid(images[0], nrow=batch_size // 5, normalize=True)

    # convert from torch tensor to numpy for plotting
    np_img = img_grid.permute(1, 2, 0).numpy()

    # plot
    plt.figure(figsize=(16, 8))
    plt.imshow(np_img)
    plt.title("Batch of Augmented Images")
    plt.axis('off')

    return plt.show()

def save_checkpoint(state, path):
    """Save specifed 'state' package to designated path.
    """
    torch.save(state, path)
