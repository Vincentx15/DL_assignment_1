import os

from os.path import isfile, join, isdir

from torch.utils.data.dataset import Dataset

import pandas as pd

from PIL import Image


class KaggleDataset(Dataset):
    conversion = {
        'Cat': 0,
        'Dog': 1
    }

    def __init__(self, data_dir, transform=None):
        """
        Instantiation of the torch Dataset abstract class, adapted to the project.
        The images are treated in PIL format.

        Args:
            data_dir (str): directory of the stored images.
            transform: a transformation to be applied on the samples.
        """

        path = os.path.abspath(data_dir)

        folders = [
            f for f in os.listdir(path) if isdir(join(path, f))
        ]

        metadata = []

        for folder in folders:
            path = data_dir + folder
            samples = pd.DataFrame([f for f in os.listdir(
                path) if isfile(join(path, f))], columns=['filename'])
            samples['folder'] = folder
            metadata.append(samples)

        self.metadata = pd.concat(metadata, axis=0)

        # Retrieve the id for each image (there are duplicates during training)
        self.metadata['id'] = self.metadata['filename'].apply(
            lambda x: int(x.split('.')[0]))

        # Sort images depending on their id
        self.metadata.sort_values(['id'], inplace=True)
        self.metadata.reset_index(inplace=True, drop=True)

        self.data_dir = os.path.abspath(data_dir)

        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, item):

        image_path, folder, id_image = self.metadata.loc[item]

        image = Image.open(
            join(self.data_dir, folder, image_path)).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        sample = {
            'image': image,
            'id': id_image,
            'label': self.conversion.get(folder, -1)
        }

        return sample
