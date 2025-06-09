#%% Imports

import torch
from collections import Counter

#%% Methods

def weighted_sampling(dataset):
    """Weighted sampling method to counteract unevenly represented classes.
    """
    class_count = Counter(dataset.targets) # count no of samples per class

    # assign weight (i.e., sampling probabuility) proportional to inverse freqauncy (rarer class = high weight)
    class_weights = {cls: 1.0 / count for cls, count in class_count.items()}

    # assign class weights to each individual sample accordingly
    sample_weights = [class_weights[label] for label in dataset.targets]
    sample_weights = torch.DoubleTensor(sample_weights) # convert to pytorch tensor for sampler

    # create sampler (replacment required for oversampling)
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    return sampler