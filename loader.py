import torch
import numpy as np
import torch.utils.data


class FERDataReader(torch.utils.data.Dataset):
    def __init__(self, mode='train'):
        super(FERDataReader, self).__init__()

        self.input_image_array = torch.from_numpy(np.load('data/{}_images.npy'.format(mode)))
        self.target_image_class = torch.from_numpy(np.load('data/{}_labels.npy'.format(mode)))

        
    def __getitem__(self, index):
        return self.input_image_array[index], self.target_image_class[index]
        
    
    def __len__(self):
        return len(self.input_image_array)