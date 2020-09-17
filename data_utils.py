import os
import json
import pickle
import numpy as np
from torch.utils.data import Dataset

class Vocab:
    ''' vocabulary of dataset '''
    def __init__(self, vocab_list, add_pad=True, add_unk=True):
        self._vocab_dict = dict()
        self._reverse_vocab_dict = dict()
        self._length = 0
        if add_pad: # pad_id should be zero (for mask)
            self.pad_word = '<pad>'
            self.pad_id = self._length
            self._vocab_dict[self.pad_word] = self.pad_id
            self._length += 1
        if add_unk:
            self.unk_word = '<unk>'
            self.unk_id = self._length
            self._vocab_dict[self.unk_word] = self.unk_id
            self._length += 1
        for w in vocab_list:
            self._vocab_dict[w] = self._length
            self._length += 1
        for w, i in self._vocab_dict.items():
            self._reverse_vocab_dict[i] = w
    
    def word_to_id(self, word):
        if hasattr(self, 'unk_id'):
            return self._vocab_dict.get(word, self.unk_id)
        return self._vocab_dict[word]
    
    def id_to_word(self, idx):
        if hasattr(self, 'unk_word'):
            return self._reverse_vocab_dict.get(idx, self.unk_word)
        return self._reverse_vocab_dict[idx]
    
    def has_word(self, word):
        return word in self._vocab_dict
    
    def __len__(self):
        return self._length
    

class Tokenizer:
    ''' transform text to indices '''
    def __init__(self, word_vocab, lower):
        ner_list = ['O', 'B', 'I']
        ner_vocab = Vocab(ner_list, add_pad=False, add_unk=False)
        self.vocab = {'word': word_vocab, 'ner': ner_vocab}
        self.maxlen = {
            'res14': {'word': 80},
            'laptop': {'word': 80},
            'res16': {'word': 80},
            'elec': {'word': 400},
            'yelp': {'word': 400},
            'res14_ae': {'word': 80, 'ner': 80},
            'laptop_ae': {'word': 80, 'ner': 80}
        }
        self.lower = lower
    
    @classmethod
    def from_files(cls, fnames, lower=True):
        all_tokens = set()
        for fname in fnames:
            fdata = json.load(open(fname, 'r', encoding='utf-8'))
            for data in fdata:
                all_tokens.update([w.lower() if lower else w for w in data['token']])
        all_tokens.update(['<aspect>', '</aspect>'])
        return cls(word_vocab=Vocab(all_tokens), lower=lower)
    
    @staticmethod
    def _pad_sequence(sequence, pad_id, maxlen, dtype='int64', padding='post', truncating='post'):
        x = (np.zeros(maxlen) + pad_id).astype(dtype)
        if truncating == 'pre':
            trunc = sequence[-maxlen:]
        else:
            trunc = sequence[:maxlen]
        trunc = np.asarray(trunc, dtype=dtype)
        if padding == 'post':
            x[:len(trunc)] = trunc
        else:
            x[-len(trunc):] = trunc
        return x
    
    def to_sequence(self, tokens, vocab_name, domain, reverse=False, padding='post', truncating='post'):
        if vocab_name == 'word' and self.lower:
            tokens = [t.lower() for t in tokens]
        sequence = [self.vocab[vocab_name].word_to_id(t) for t in tokens]
        pad_id = self.vocab[vocab_name].pad_id if hasattr(self.vocab[vocab_name], 'pad_id') else 0
        maxlen = self.maxlen[domain][vocab_name]
        if reverse:
            sequence.reverse()
        return Tokenizer._pad_sequence(sequence, pad_id=pad_id, maxlen=maxlen, padding=padding, truncating=truncating)
    
    def pad_sequence(self, sequence, pad_id, maxlen, reverse=False, dtype='int64', padding='post', truncating='post'):
        if dtype == 'int64':
            sequence = [int(w) for w in sequence]
        elif dtype == 'float32':
            sequence = [float(w) for w in sequence]
        if reverse:
            sequence.reverse()
        return Tokenizer._pad_sequence(sequence, pad_id=pad_id, maxlen=maxlen, dtype=dtype, padding=padding, truncating=truncating)
    
class MyDataset(Dataset):
    ''' PyTorch standard dataset class '''
    def __init__(self, side, tasks, domains, fname, tokenizer):
        data_file = os.path.join('dats', '{:s}_{:s}'.format('_'.join(domains.values()), os.path.split(fname)[-1].replace('.json', '.cache')))
        if os.path.exists(data_file):
            print('loading dataset: {:s}'.format(data_file))
            dataset = pickle.load(open(data_file, 'rb'))
        else:
            print('building dataset...')
            read_func = {'asc': self._read_asc, 'dsc': self._read_dsc, 'ae': self._read_ae}
            dataset = read_func[tasks[side]](fname, domains[side], tokenizer)
            pickle.dump(dataset, open(data_file, 'wb'))
        self._dataset = dataset
    
    @staticmethod
    def _read_asc(fname, domain, tokenizer):
        polarity_dict = {'positive': 0, 'negative': 1, 'neutral': 2}
        dataset = list()
        fdata = json.load(open(fname, 'r', encoding='utf-8'))
        for data in fdata:
            for aspect in data['aspects']:
                start, end = int(aspect['from']), int(aspect['to'])
                text = data['token'][:start] + ['<aspect>'] + data['token'][start:end] + ['</aspect>'] + data['token'][end:]
                text = tokenizer.to_sequence(text, 'word', domain)
                term = tokenizer.to_sequence(aspect['term'], 'word', domain)
                aspect_mask = [1 if start <= i < end else 0 for i in range(len(data['token']))]
                aspect_mask = tokenizer.pad_sequence(aspect_mask, 0, maxlen=tokenizer.maxlen[domain]['word'])
                polarity = polarity_dict[aspect['polarity']]
                dataset.append({'text': text, 'aspect': term, 'aspect_mask': aspect_mask, 'polarity': polarity})
        return dataset
    
    @staticmethod
    def _read_dsc(fname, domain, tokenizer):
        polarity_dict = {'positive': 0, 'negative': 1, 'neutral': 2}
        dataset = list()
        fdata = json.load(open(fname, 'r', encoding='utf-8'))
        for data in fdata:
            text = tokenizer.to_sequence(data['token'], 'word', domain)
            term = tokenizer.to_sequence(['NULL'], 'word', domain)
            aspect_mask = tokenizer.pad_sequence([], 0, maxlen=tokenizer.maxlen[domain]['word'])
            polarity = polarity_dict[data['polarity']]
            dataset.append({'text': text, 'aspect': term, 'aspect_mask': aspect_mask, 'polarity': polarity})
        return dataset
    
    @staticmethod
    def _read_ae(fname, domain, tokenizer):
        dataset = list()
        fdata = json.load(open(fname, 'r', encoding='utf-8'))
        for data in fdata:
            text = tokenizer.to_sequence(data['token'], 'word', domain)
            term = tokenizer.to_sequence(['NULL'], 'word', domain)
            aspect_mask = tokenizer.pad_sequence([], 0, maxlen=tokenizer.maxlen[domain]['word'])
            polarity = tokenizer.to_sequence(data['ner'], 'ner', domain) # ner labels
            dataset.append({'text': text, 'aspect': term, 'aspect_mask': aspect_mask, 'polarity': polarity})
        return dataset
    
    def __getitem__(self, index):
        return self._dataset[index]
    
    def __len__(self):
        return len(self._dataset)

def build_tokenizer(domains, fnames):
    data_file = os.path.join('dats', f"{'_'.join(domains.values())}_tokenizer.dat")
    if os.path.exists(data_file):
        print(f"loading tokenizer: {data_file}")
        tokenizer = pickle.load(open(data_file, 'rb'))
    else:
        print('building tokenizer...')
        tokenizer = Tokenizer.from_files(fnames)
        pickle.dump(tokenizer, open(data_file, 'wb'))
    return tokenizer

def _load_wordvec(data_path, word_dim, vocab=None):
    with open(data_path, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
        word_vec = dict()
        word_vec['<pad>'] = np.zeros(word_dim).astype('float32')
        for line in f:
            tokens = line.rstrip().split()
            if (len(tokens)-1) != word_dim:
                continue
            if tokens[0] == '<pad>' or tokens[0] == '<unk>': # avoid them
                continue
            if vocab is None or vocab.has_word(tokens[0]):
                word_vec[tokens[0]] = np.asarray(tokens[1:], dtype='float32')
        return word_vec

def build_embedding_matrix(domains, vocab, word_dim=300):
    data_file = os.path.join('dats', f"{'_'.join(domains.values())}_embedding_matrix.dat")
    if os.path.exists(data_file):
        print(f"loading embedding matrix: {data_file}")
        embedding_matrix = pickle.load(open(data_file, 'rb'))
    else:
        print('loading word vectors...')
        embedding_matrix = np.random.uniform(-0.25, 0.25, (len(vocab), word_dim)).astype('float32')
        word_vec = _load_wordvec(os.path.join('glove', 'glove.840B.300d.txt'), word_dim, vocab)
        for i in range(len(vocab)):
            vec = word_vec.get(vocab.id_to_word(i))
            if vec is not None:
                embedding_matrix[i] = vec
        pickle.dump(embedding_matrix, open(data_file, 'wb'))
    return embedding_matrix
