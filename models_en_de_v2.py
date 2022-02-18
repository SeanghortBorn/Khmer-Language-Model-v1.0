import random
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Open Data
path = 'data/'
f = open(path + 'SBBICkm_KH.txt', 'r', encoding="utf8")
all_words = f.readlines()
f.close()

# Defind Khmer Characters
UNI_KA = 0x1780
UNI_LAST = 0x17ff
C_START = UNI_LAST - UNI_KA + 1
C_STOP = C_START + 1
C_UNK = C_START + 2
N_CHAR = UNI_LAST - UNI_KA + 1 + 3

# Create functions to Generate Error Words Pairs
def str_insert(str):
    pos = random.randint(0, len(str))
    rand_c = chr(UNI_KA + random.randint(0, N_CHAR - 1 - 3))
    return str[:pos] + rand_c + str[pos:]

def str_delete(str):
    pos = random.randint(0, len(str) - 1)
    return str[:pos] + str[pos + 1:]

def str_replace(str):
    pos = random.randint(0, len(str) - 1)
    rand_c = chr(UNI_KA + random.randint(0, N_CHAR - 1 - 3))
    return str[:pos] + rand_c + str[pos + 1:]

def str_rand_err(str):
    t = random.randint(0, 2)
    if t == 0:
        return str_insert(str)
    elif t == 1:
        return str_delete(str)
    else:
        return str_replace(str)


def str2ints(str):
    tmp = []
    for c in str:
        if ord(c) < UNI_KA or ord(c) > UNI_LAST:
            c = C_UNK
        else:
            c = ord(c) - UNI_KA
        tmp.append(c)
    return tmp

def onehot(ints, n_class):
    """
  ints: np (l) of int
  n_class: int
  return: np (l, n_class) of int
  """
    l = len(ints)
    tmp = np.zeros((l, n_class), dtype=int)
    for j, i in enumerate(ints):
        tmp[j, i] = 1
    return tmp

def input0_tensor(b_sz):
    tmp = np.zeros((b_sz, N_CHAR), dtype=int)
    tmp[:, C_START] = 1
    return torch.tensor(tmp, dtype=torch.float32)

# Create models to train and test data: Encoder and Decoder
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

def word2tensor(ws):
    """
    ws: list (b) of string
    return: tensor (b, max_len_str)
    """
    tmp = []
    max_len = 0
    for w in ws:
        if len(w) > max_len:
            max_len = len(w)
        tmp.append(str2ints(w))
    np_tmp = np.ones((len(ws), max_len + 1), dtype=int) * C_UNK
    for i in range(len(ws)):
        np_tmp[i, :len(tmp[i])] = np.array(tmp[i], dtype=int)
        np_tmp[i, len(tmp[i])] = C_STOP
    # print(np_tmp)
    np_tmp = np_tmp.flatten()
    np_tmp = onehot(np_tmp, N_CHAR)
    np_tmp = np_tmp.reshape((len(ws), max_len + 1, -1))
    return torch.tensor(np_tmp, dtype=torch.float32)


def label2tensor(ws):
    tmp = []
    max_len = 0
    for w in ws:
        if len(w) > max_len:
            max_len = len(w)
        tmp.append(str2ints(w))
    np_tmp = np.ones((len(ws), max_len + 1), dtype=int) * C_UNK
    for i in range(len(ws)):
        np_tmp[i, :len(tmp[i])] = np.array(tmp[i], dtype=int)
        np_tmp[i, len(tmp[i])] = C_STOP
    t_tmp = torch.tensor(np_tmp, dtype=torch.long)

    coef = np.ones((len(ws), max_len + 1))
    y_len = []
    for i in range(len(ws)):
        coef[i, len(ws[i]) + 1:] = 0
        y_len.append(len(ws[i]) + 1)

    return t_tmp, torch.tensor(coef, dtype=torch.float32), torch.tensor(y_len, dtype=torch.float32)

def tensor2str(predict):
    tmp = predict.numpy()
    lst_s = []
    for i in range(tmp.shape[0]):
        s = ''
        for c in tmp[i]:
            if c == C_STOP:
                break
            if c >= 0 and c <= UNI_LAST - UNI_KA:
                s += chr(UNI_KA + c)
        lst_s.append(s)
    return lst_s

words = []
for w in all_words:
    w = w[:-1]
    if len(w) >= 5 and len(w) <= 6:
        to_add = True
        for c in w:
            if ord(c) < UNI_KA or ord(c) > UNI_LAST:
                to_add = False
                break
        if to_add:
            words.append(w)

# Prepare data for training: Select 'numbers_of_words' word from the original dataset 'SBBIC.txt' randomly
# numbers_of_words = 1000
# random.shuffle(words)
random.shuffle(all_words)
# words = words[:numbers_of_words]

f = open(path + 'kh_words.txt', 'w', encoding="utf8")
for w in words:
    f.write(w + '\n')
f.close()

# Train model
# Define Parameters
b_sz = 100
hidden_size = 512
MAX_LEN = 10
lr = 0.001
# MAX_LEN = len(word)

hid_sz = hidden_size
encoder = Encoder(hid_sz)
decoder = Decoder(hid_sz, MAX_LEN)

cost_fn = nn.CrossEntropyLoss(reduction='none')
opt = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=lr)

# words = words[:100]+words[8000:8100]+words[-100:]
print("\nTraining dataset size:" ,len(words),"words\n")

for j in range(1000):
    random.shuffle(words)
    i = 0
    while i + b_sz <= len(words):
        y = words[i:i + b_sz]
        x = []
        for _ in y:
            x.append(str_rand_err(_))

        t_y, coef, y_len = label2tensor(y)  # tensor (b, len, in_sz)
        t_x = word2tensor(x)  # tensor (b, len, in_sz)
        t_x = t_x.permute(1, 0, 2)  # tensor (len, b, in_sz)

        opt.zero_grad()
        hid = encoder(t_x)
        outputs = decoder(hid)  # (b, n_char, max_len)

        loss = cost_fn(outputs[:, :, :t_y.size(1)], t_y)
        loss = (loss * coef).sum(dim=1) / y_len
        loss = loss.mean()
        if j % 10 == 0 and i==0:
            with torch.no_grad():
                print('epoch', j, 'iter', i, ':', loss.item())
                predict = torch.argmax(outputs, dim=1)
                print(x)
                print(tensor2str(predict))
                print(y)
                print()

            torch.save(encoder.state_dict(), path + 'encoder.sav')
            torch.save(decoder.state_dict(), path + 'decoder.sav')
            torch.save(opt.state_dict(), path + 'optimizer.sav')

        loss.backward()
        opt.step()
        i += b_sz

# Test model
f = open(path + 'kh_words.txt', 'r', encoding="utf8")
_all_words = f.readlines()
f.close()
_words = []
for w in _all_words:
    _words.append(w[:-1])

hid_sz = hidden_size
encoder = Encoder(hid_sz)
decoder = Decoder(hid_sz, MAX_LEN)

encoder.load_state_dict(torch.load(path + 'encoder.sav'))
decoder.load_state_dict(torch.load(path + 'decoder.sav'))

encoder.eval()
decoder.eval()

i = 0
b_sz = 50
n_correct = 0
while i + b_sz <= len(_words):
    y = _words[i:i + b_sz]
    x = []
    for _ in y:
        x.append(str_rand_err(_))

    t_y, coef, y_len = label2tensor(y)  # tensor (b, len, in_sz)
    t_x = word2tensor(x)  # tensor (b, len, in_sz)
    t_x = t_x.permute(1, 0, 2)  # tensor (len, b, in_sz)

    hid = encoder(t_x)
    outputs = decoder(hid)  # (b, n_char, max_len)

    predict = torch.argmax(outputs, dim=1)
    z = tensor2str(predict)

    for n in range(len(y)):
        if y[n] == z[n]:
            n_correct += 1
    i += b_sz

print('ACCURACY: %.2f%%' % (n_correct * 100.0 / len(_words)))
