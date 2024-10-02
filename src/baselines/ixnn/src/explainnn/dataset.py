from torchvision import datasets
import torchvision.transforms as transforms

from explainnn.definitions import DATASET_DIR, SUPPORTED_DATASET

def get_dataset(dataset_name):
    """
    Load dataset with the correct transformations

    Input
    -------
    dataset_name : {'MNIST', 'CIFAR10'}
        The name of the dataset to load

    Outputs
    -------
    train_set :

    validation_set :

    """
    if dataset_name not in SUPPORTED_DATASET:
        raise ValueError(f"""
    Dataset "{dataset_name}" not supported.
    Currently supported datasets are :
        {', '.join(SUPPORTED_DATASET)}
                        """)
    imsize = 32
    train_set = datasets.__dict__[dataset_name](root=DATASET_DIR.joinpath(dataset_name), train=True, 
                                            transform=_get_transforms(dataset_name, imsize), download=True)

    validation_set = datasets.__dict__[dataset_name](root=DATASET_DIR.joinpath(dataset_name), train=False, 
                                            transform=_get_transforms(dataset_name, imsize), download=True)

    return train_set, validation_set

def _get_transforms(dataset_name, imsize):
    """Return the transform set associated to dataset_name"""
    if dataset_name == 'MNIST':
        return transforms.Compose([transforms.Resize((imsize, imsize)),
                                 transforms.Grayscale(num_output_channels=1), #3 for resnet and 1 for LeNet
                                 transforms.ToTensor(),
                                ])
    elif dataset_name == 'CIFAR10':
        return transforms.Compose([transforms.Resize((imsize, imsize)),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), 
                                ])
    elif dataset_name == 'ImageNet':
        return transforms.Compose([
                transforms.Resize((imsize,imsize)),
                transforms.CenterCrop((imsize,imsize)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])]),

        return transforms.Compose([transforms.Resize((imsize, imsize)),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), 
                                ])
    
    else:
        raise ValueError(f"""
    Dataset "{dataset_name}" not supported.
    Currently supported datasets are :
        {', '.join(SUPPORTED_DATASET)}
                        """)
