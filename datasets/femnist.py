import numpy as np

from typing import Any
from torch.utils.data import Dataset
import datasets.np_transforms as nptr

IMAGE_SIZE = 28

class Femnist(Dataset):

    def __init__(self, data: dict,transform: nptr.Compose,client_name: str, angle,Spectre=False,ls=None):
        super().__init__()
        self.samples = [(image, label) for image, label in zip(data['x'], data['y'])]
        self.transform = transform
        self.client_name = client_name
        if Spectre:
            self.transform= nptr.Compose([self.transform, nptr.Rotate(angle),nptr.SpecTran(dim=(-2,-1)),nptr.StyleRem(ls)])
        else:
            self.transform = nptr.Compose([self.transform, nptr.Rotate(angle)])


    def __getitem__(self, index: int) -> Any:
        # returns image and label corresponding to the index
        image, label = self.samples[index]
        numpy_img = np.array(image).reshape(28, 28, 1)
        transformed_img = self.transform(numpy_img)
        return transformed_img, label

    def __len__(self) -> int:
        return len(self.samples)
