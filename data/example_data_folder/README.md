## Data

The dataset used during development was proprietary or task-specific, so it is **not included**.

The code is fully modular and includes built-in image resizing to **224×224 pixels**, the input size recommended by the ResNet18 backbone.

You can test the implementation easily with **ImageNet** or popular subsets like **Tiny ImageNet**.

For smaller-scale experiments, datasets like **CIFAR-10**, **STL-10**, or your own images can be used and will be resized automatically. Images must be prepared in the expected folder structure. The backbone is modular and easily exchangeable.

An example folder structure is provided in `example_data_folder/` illustrating the expected `.tar` or `.tar.gz` archive format.

See the `get_data` method in [`data_aug/contrastive_data.py`](simclr-small-dataset/data_aug/contrastive_data.py) for details on dataset integration.