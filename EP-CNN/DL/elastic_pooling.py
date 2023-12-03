import torch
import torch.nn as nn

class ElasticPooling(nn.Module):
    def __init__(self, p1=128, wordvec=16):
        super(ElasticPooling, self).__init__()
        self.p1 = p1
        self.wordvec = wordvec

    def forward(self, x):
        num_rows = x.shape[2] // self.p1

        pooled_values = self._get_pooled_values(x, num_rows)
        out = torch.cat(pooled_values, dim=1)
        outs = out.view(x.shape[0], self.p1, self.wordvec, x.shape[1])

        return outs

    def _get_pooled_values(self, x, num_rows):
        pooled_values = []
        for i in range(self.p1):
            for j in range(self.wordvec):
                pooling_area = self._get_pooling_area(x, i, j, num_rows)
                pooled_val, _ = torch.max(pooling_area, dim=(2, 3))
                pooled_values.append(pooled_val)
        return pooled_values

    def _get_pooling_area(self, x, i, j, num_rows):
        return x[:, :, i * num_rows:(i + 1) * num_rows, j * 2:(j + 1) * 2]
