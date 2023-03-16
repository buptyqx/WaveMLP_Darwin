import torch.onnx
import numpy as np
import scipy.sparse
from torch import nn
import torch
from torchsummary import summary
from sklearn import svm
import warnings
from wavemlp3D import WaveMLP
warnings.filterwarnings('ignore')

class WaveRank(nn.Module):
    def __init__(self, rank_pooling_C):
        super(WaveRank, self).__init__()
        self.wave_mlp = WaveMLP('M', num_classes=101)
        self.rank_pooling_C = rank_pooling_C

    def forward(self, x):
        # Wave MLP
        features = self.wave_mlp(x)
        seq = features.detach().cpu().numpy()
        rank_coef = self.rank_pooling(seq, self.rank_pooling_C)
        rank_coef = torch.from_numpy(rank_coef).to(x.device).float()
        # Weighted sum of features
        out = (features + rank_coef.unsqueeze(-1).unsqueeze(-1)).sum(dim=1)
        return out

    def smoothSeq(self, seq):
        res = np.cumsum(seq, axis=1)
        seq_len = np.size(res, 1)
        res = res / np.expand_dims(np.linspace(1, seq_len, seq_len), 0)
        return res

    def rootExpandKernelMap(self, data):
        element_sign = np.sign(data)
        nonlinear_value = np.sqrt(np.fabs(data))
        return np.vstack((nonlinear_value * (element_sign > 0), nonlinear_value * (element_sign < 0)))

    def getNonLinearity(self, data, nonLin='ref'):
        if nonLin == 'none':
            return data
        if nonLin == 'ref':
            return self.rootExpandKernelMap(data)
        elif nonLin == 'tanh':
            return np.tanh(data)
        elif nonLin == 'ssr':
            return np.sign(data) * np.sqrt(np.fabs(data))
        else:
            raise ("We don't provide {} non-linear transformation".format(nonLin))

    def normalize(self, seq, norm='l2'):
        if norm == 'l2':
            seq_norm = np.linalg.norm(seq, ord=2, axis=0)
            seq_norm[seq_norm == 0] = 1
            seq_norm = seq / np.expand_dims(seq_norm, 0)
            return seq_norm
        elif norm == 'l1':
            seq_norm = np.linalg.norm(seq, ord=1, axis=0)
            seq_norm[seq_norm == 0] = 1
            seq_norm = seq / np.expand_dims(seq_norm, 0)
            return seq_norm
        else:
            raise ("We only provide l1 and l2 normalization methods")

    def rank_pooling(self, time_seq, C, NLStyle='ssr'):
        seq_smooth = self.smoothSeq(time_seq)
        seq_nonlinear = self.getNonLinearity(seq_smooth, NLStyle)
        seq_norm = self.normalize(seq_nonlinear)
        seq_len = np.size(seq_norm, 1)
        Labels = np.array(range(1, seq_len + 1))
        seq_svr = scipy.sparse.csr_matrix(np.transpose(seq_norm))
        svr_model = svm.LinearSVR(epsilon=0.1,
                                  tol=0.001,
                                  max_iter=10000,
                                  C=C,
                                  loss='squared_epsilon_insensitive',
                                  fit_intercept=False)
        svr_model.fit(seq_svr, Labels)
        coef = svr_model.coef_
        rank_coef = np.abs(coef) / np.sum(np.abs(coef) + 1e-10)
        return rank_coef


def main():
    batch_size = 4
    model = WaveRank(rank_pooling_C=1)
    model.to('cuda')
    summary(model, input_size=(3, 16, 112, 112), batch_size=batch_size)
    x = torch.randn(batch_size, 3, 16, 112, 112).to('cuda')
    print(x)
    y = model(x)
    print(y.shape)


if __name__ == "__main__":
    main()
