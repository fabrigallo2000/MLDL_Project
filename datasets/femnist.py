import numpy as np
import datasets.np_transforms as tr

from typing import Any
from torch.utils.data import Dataset
import torch

IMAGE_SIZE = 28

#transform = torch.transforms.Compose([
#    torch.transforms.ToTensor(),
#    torch.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
# ])


class Femnist(Dataset):

    def __init__(self,
                 data: dict,
                 transform: tr.Compose,
                 client_name: str):
        super().__init__()
        self.samples = [(image, label) for image, label in zip(data['x'], data['y'])]
        # self.samples è lista di tuple [([pixel immagine], label), ()]
        self.transform = transform
        self.client_name = client_name

    def __getitem__(self, index: int) -> Any:
        # chiamata da torch, torna immagine e label corrispondenti all'indice
        image, label = self.samples[index]
        tensor_img = torch.tensor(image)
        return tensor_img.reshape(1, 28, 28), label

    def __len__(self) -> int:
        return len(self.samples)
