import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, hid_sz):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(input_size=N_CHAR, hidden_size=hid_sz)

    def forward(self, x):
        """
        x: tensor (len, b, in_sz)
        return last hid state: tensor (b, hid_sz)
        """
        output, (h_n, c_n) = self.lstm(x)
        return torch.squeeze(h_n)

class Decoder(nn.Module):
    def __init__(self, hid_sz, max_len):
        super(Decoder, self).__init__()
        self.rnn = nn.LSTMCell(input_size=N_CHAR + hid_sz, hidden_size=hid_sz)
        self.out = nn.Linear(hid_sz, N_CHAR)
        self.hid_sz = hid_sz
        self.max_len = max_len

    def forward(self, hid):
        """
        hid: tensor (b, hid_sz)
        return tensor (b, n_char, max_len)
        """
        input = input0_tensor(b_sz)
        cell = torch.randn(b_sz, self.hid_sz)
        outputs = []
        for _ in range(self.max_len):
            hid, cell = self.rnn(torch.cat((input, hid), dim=1), (hid, cell))
            outputs.append(self.out(hid))
        return torch.stack(outputs, dim=2)
