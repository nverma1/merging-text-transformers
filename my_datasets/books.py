import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer
from functools import partial


CONTEXT_LEN=512
class Books(Dataset):
    def __init__(self,
                 root=None,
                 train=True,
                 sort=False,
                 bpe='bert',
                 num=0): # do bpe

        if root:
            self.root = root
        else:
            self.root = f'my_datasets/sample{num}.txt'
        self.name = 'books'
        self.train = train  
        self.sort = sort
        self.bpe = bpe

        split = self.load_split() # loads split & labels

        self.dataset = split
        if bpe == 'bert':
            self.bert_path = 'google/multiberts-seed_0'
            self.bert_encoder = BertTokenizer.from_pretrained(self.bert_path)


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (encoded, len) 
        """

        sentence = self.dataset[index]
        if self.bpe == 'bert':
            tokens = self.bert_encoder(sentence, return_tensors='pt')['input_ids'][0]
        # automatically truncating
        if len(tokens) > CONTEXT_LEN:
            return tokens[:CONTEXT_LEN], CONTEXT_LEN

        return tokens, len(tokens) # sequence of tokens, None label as placeholder

    def load_split(self):

        if self.train:
            # skip first header line
            train_lines = open(self.root).readlines()
            cleaned_lines = [line.strip() for line in train_lines]
            
            if self.sort:
                cleaned_lines_sorted = sorted(cleaned_lines, key=len)
                return cleaned_lines_sorted
        return cleaned_lines

def pad_collate(batch, pad_tok):
    (xx, lens) = zip(*batch)
    xx_pad = pad_sequence(xx, batch_first=True, padding_value=pad_tok)
    return xx_pad, lens

def prepare_train_loaders(config):
    if config['tokenizer'] == 'bert':
        pad_tok = 0
    elif config['tokenizer'] == 'roberta':
        pad_tok = 1
    return {
        'full': torch.utils.data.DataLoader(
            Books( train=True, 
                num=config['num'], 
                sort=config['sorted'], 
                bpe=config['tokenizer']),
                batch_size=config['batch_size'],
            shuffle=True, collate_fn=partial(pad_collate, pad_tok=pad_tok)
        )
    }

def prepare_test_loaders(config):
    if config['tokenizer'] == 'bert':
        pad_tok = 0
    elif config['tokenizer'] == 'roberta':
        pad_tok = 1
    return {
        'full': torch.utils.data.DataLoader(
            Books( train=False),
            batch_size=config['batch_size'],
            shuffle=True, num=0,
            num_workers=config['num_workers'], collate_fn=partial(pad_collate, pad_tok=pad_tok)
        )
    }

if __name__ == "__main__":
    config = {'batch_size': 4, 'num_workers': 4, 'shuffle_train': True, 'sorted': False, 'tokenizer': 'bert', 'num': 0}
    train_loader = prepare_train_loaders(config)
    x = next(iter(train_loader['full']))
    print(x)