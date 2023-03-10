import torch
import numpy as np
from tqdm.auto import tqdm
from functions import load_scan, get_pixels_hu, resample, segment_lung_mask, pad

class LungCT_Dataset(torch.utils.data.Dataset):

    def __init__(self, dataPath, status, size=128):
        self.labels = []
        self.paths = []
        images = []
        data = []

        tqdm.write('start loading dataset...')

        for x in tqdm(status):
            image = load_scan(dataPath + x[0].replace('Patient', 'R'))

            if image is not None:
                self.paths.append(x[0])
                self.labels.append(x[1])

                pixels = get_pixels_hu(image)
                scaled = resample(pixels, image, size)
                segmented = segment_lung_mask(scaled)

                padded = pad(segmented, size)

                images.append(padded)
                data.append(torch.Tensor(padded).unsqueeze(0).float())

        self.images = np.stack(images)
        self.data = torch.stack(data).float()

        tqdm.write('dataset loaded successfully')

    def __getitem__(self, index):
        return self.data[index], self.images[index], self.labels[index], self.paths[index]

    def __len__(self):
        return len(self.labels)