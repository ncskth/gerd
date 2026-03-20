import torch
from torch.utils.data import Dataset

class SequenceChunkDataset(Dataset):
    def __init__(self, file_paths, start_idx=0, datapoint_length=40, stride = None):
        self.file_paths = file_paths
        self.start_idx = start_idx
        self.datapoint_length = datapoint_length
        self.stride = stride or datapoint_length
        self.index_map = []  # (file_idx, start_t)

        # Build index mapping
        for file_idx, path in enumerate(file_paths):
            data, _ = torch.load(path)  # only to get length
            T = data.shape[0]
            t = start_idx
            while t + datapoint_length <= T:
                self.index_map.append((file_idx, t))
                t += self.stride  # non-overlapping chunks

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        file_idx, start_t = self.index_map[idx]
        path = self.file_paths[file_idx]

        data, labels = torch.load(path)

        data = data.to_dense().cpu()

        end_t = start_t + self.stride

        x = data[start_t:end_t]
        y = labels[start_t:end_t]

        return x, y