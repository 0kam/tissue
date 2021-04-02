from torch import nn
import torch
from torch.nn import functional as F

from pixyz.distributions import Normal, RelaxedCategorical, Deterministic

## inference model q(z|x,y)
class Inference(Normal):
    def __init__(self, x_dim, z_dim, y_dim, drop_out_rate):
        super().__init__(cond_var=["x","y"], var=["z"], name="q")

        self.h_dim = int((x_dim + z_dim) / 2)
        self.prelu1 = nn.PReLU()
        self.do1 = nn.Dropout()
        self.lstm = nn.LSTM(input_size=x_dim+y_dim, hidden_size=self.h_dim, batch_first=True)
        
        self.fc11 = nn.Linear(self.h_dim, z_dim)
        self.fc12 = nn.Linear(self.h_dim, z_dim)

    def forward(self, x, y):
        y = y.repeat(x.shape[1], 1, 1).transpose(0,1)
        h = torch.cat([x, y], 2)
        _, h = self.lstm(h)
        h = h[0].view(-1, self.h_dim)
        return {"loc": self.fc11(h), "scale": F.softplus(self.fc12(h))}

# generative model p(x|z) 
class Generator(Normal):
    def __init__(self, x_dim, z_dim, seq_length, device):
        super().__init__(cond_var=["z"], var=["x"], name="p")

        h_dim = int((x_dim + z_dim) / 2)
        self.device = device
        self.seq_length = seq_length
        self.fc1 = nn.Linear(z_dim, h_dim)
        self.tanh = nn.Tanh()
        self.lstm = nn.LSTM(h_dim, h_dim, batch_first=True)
        self.fc21 = nn.Linear(h_dim, x_dim)
        self.fc22 = nn.Linear(h_dim, x_dim)

    def forward(self, z):
        h = self.tanh(self.fc1(z))
        h = h.unsqueeze(1)
        x1 = torch.zeros(h.shape).to(self.device)
        cell_state = torch.zeros(h.shape).transpose(0,1).to(self.device)
        init_hidden = h.transpose(0,1)
        x = []
        for _ in range(self.seq_length):
            _, hc = self.lstm(x1, (init_hidden, cell_state))
            x1 = hc[0].transpose(0,1)
            x.append(x1)
        x = torch.stack(x, dim=1).squeeze()

        return {"loc": self.fc21(x), "scale": F.softplus(self.fc22(x))}

# classifier p(y|x)
class Classifier(RelaxedCategorical):
    def __init__(self, x_dim, y_dim, drop_out_rate):
        super().__init__(cond_var=["x"], var=["y"], name="p")
        self.h_dim = int((x_dim + y_dim) / 2)

        self.do1 = nn.Dropout(drop_out_rate)
        self.lstm = nn.LSTM(input_size=x_dim, hidden_size=self.h_dim, batch_first=True)
        self.prelu1 = nn.PReLU()
        self.fc2 = nn.Linear(self.h_dim, y_dim)

    def forward(self, x):
        x = self.do1(x)
        _, h = self.lstm(x)
        h = h[0].view(-1, self.h_dim)
        h = F.softmax(self.fc2(h), dim=1)
        h = h + 1e-9
        return {"probs": h}


# prior model p(z|y)
class Prior(Normal):
    def __init__(self, z_dim, y_dim):
        super().__init__(var=["z"], cond_var=["y"], name="p_{prior}")

        self.fc11 = nn.Linear(y_dim, z_dim)
        self.fc12 = nn.Linear(y_dim, z_dim)
    
    def forward(self, y):
        return {"loc": self.fc11(y), "scale": F.softplus(self.fc12(y))}
