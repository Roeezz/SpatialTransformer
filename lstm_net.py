import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(82, 82, num_layers=1)

        # convolution for the pictures
        ngf = 128
        self.encoder = nn.Sequential(
            nn.Conv2d(51, ngf, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2),

            nn.Conv2d(ngf, ngf * 2, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2),

            nn.Conv2d(ngf * 2, ngf * 4, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(True),
            nn.Conv2d(ngf * 4, ngf * 4, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2),

            # nn.Conv2d(ngf * 4, ngf * 4, kernel_size=(3, 3), stride=1),
            # nn.BatchNorm2d(ngf * 4),
            # nn.LeakyReLU(True),
            # nn.Conv2d(ngf * 8, ngf * 8, kernel_size=(3, 3), stride=1),
            # nn.BatchNorm2d(ngf * 8),
            # nn.LeakyReLU(True),
            # nn.MaxPool2d(kernel_size=(3, 3), stride=2),

            nn.Conv2d(ngf * 4, 500, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(500),
            nn.LeakyReLU(True),
            nn.Conv2d(500, 500, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(500),
            nn.LeakyReLU(True),

            nn.AvgPool2d(kernel_size=(17, 32), stride=(17, 32))
        )

        self.fc_enc = nn.Sequential(
            nn.Linear(500, 128),
            nn.LeakyReLU(0.2, True),
            nn.Linear(128, 82)
        )

    def forward(self, xs, xs_flow, label, label_pred):
        # transform the input
        batch_sz = label_pred.shape[1]
        seq_sz = label_pred.shape[0]
        xs = xs.reshape(batch_sz, -1, 200, 320)
        xs_flow = xs_flow.reshape(batch_sz, -1, 200, 320)
        xs = torch.cat((xs, xs_flow), dim=1)

        hidden = self.encoder(xs)
        hidden = hidden.reshape(-1, 500)
        hidden = self.fc_enc(hidden).view(1, batch_sz, 82)
        out = label
        hidden_in = (hidden, hidden)
        for i in range(seq_sz):
            out, hidden_in = self.lstm(out.view(1, batch_sz, 82), hidden_in)
            out = F.log_softmax(out, dim=1)
            label_pred[i] = out

        return label_pred
