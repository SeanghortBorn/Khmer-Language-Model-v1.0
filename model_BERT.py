import torch
import re
import torch.nn as nn

text = (
       'Hello, how are you? I am Romeo.\n'
       'Hello, Romeo My name is Juliet. Nice to meet you.\n'
       'Nice meet you too. How are you today?\n'
       'Great. My baseball team won the competition.\n'
       'Oh Congratulations, Juliet\n'
       'Thanks you Romeo'
   )
# text = ('ព្រះរាជាណាចក្រកម្ពុជា')
sentences = re.sub("[.,!?\\-]", '', text.lower()).split('\n')
word_list = list(set(" ".join(sentences).split()))
# print(word_list,sentences)
word_dict = {'[PAD]': 0, '[CLS]': 1, '[SEP]': 2, '[MASK]': 3}
for i, w in enumerate(word_list):
       word_dict[w] = i+4
       number_dict = {i: w for i, w in enumerate(word_dict)}
       vocab_size = len(word_dict)

def make_batch():
       global n_pred
       batch = []
       posititve = negative = 0
       while posititve != batch_size/2 or negative != batch_size/2:
              tokens_a_index, tokens_b_index = randrange(len(sentences)), randrange(len(sentences))
              tokens_a, tokens_b = token_list[tokens_a_index], token_list[tokens_b_index]
              input_ids = [word_dict['[CLS]']] + tokens_a + [word_dict['[SEP]']] + tokens_b + [word_dict['[SEP]']]
              segment_ids = [0] * (1 + len(tokens_a) + 1) + [1] * (len(tokens_b) + 1)

              # Mask LM
              n_pred = min(max_pred, max(1, int(round(len(input_ids) * 0.15))))
              cand_maked_pros = [i for i, token in enumerate(input_ids)
                                 if token != word_dict['[CLS]'] and token != word_dict['[SEP]']]
              shuffle(cand_maked_pros)
              masked_tokens, masked_pros = [], []
              for pos in cand_maked_pros[:n_pred]:
                     masked_pros.append(pos)
                     masked_tokens.append(input_ids[pos])
                     if random() <0.8:
                            input_ids[pos] = word_dict['[MASK]'] #make mask
                     elif random() < 0.5:
                            index = randint(0, vocab_size - 1) #random index in vocabulary
                            input_ids[pos] = word_dict[number_dict[index]] #replace

              #Zero Paddings
              n_pad = maxlen - len(input_ids)
              input_ids.extend([0]*n_pad)
              segment_ids.extend([0]*n_pad)

              #Zero padding (100%-15%) token
              if max_pred > n_pred:
                     n_pad = max_pred - n_pred
                     masked_tokens.extend([0] * n_pad)

              if tokens_a_index +1 == tokens_b_index and posititve < batch_size/2:
                     batch.append([input_ids, segment_ids, masked_tokens, masked_pros, True]) #IsNext
                     posititve += 1
              elif tokens_a_index +1 != tokens_b_index and negative < batch_size/2:
                     batch.append([input_ids, segment_ids, masked_tokens, masked_pros, False]) #NoNext
                     negative += 1
       return batch

class Embedding(nn.Module):
       def __init__(self):
              super(Embedding, self).__init__()
              self.tok_embed = nn.Embedding(vocab_size, d_model)
              self.pos_embed = nn.Embedding(maxlen, d_model)
              self.seg_embed = nn.Embedding(n_segments, d_model)
              self.norm = nn.LayerNorm(d_model)

       def forward(self, x, seg):
              seq_len = x.size(1)
              pos = torch.arange(seq_len, dtype = torch.long)
              pos = pos.unsqueeze(0).expand_as(x)
              embedding = self.tok_embed(x) + self.pos_embed(pos) + self.seg_embed(seg)
              return self.norm(embedding)

def get_attn_pad_mask(seq_q, seq_k):
       batch_size, len_q = seq_q.size()
       batch_size, len_k = seq_k.size()
       pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)
       return pad_attn_mask.expand(batch_size, len_q, len_k)

print(get_attn_pad_mask(input_ids, input_ids)[0][0], input_ids[0])
