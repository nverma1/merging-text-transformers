import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
from transformers import BertTokenizer
from functools import partial

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

CONTEXT_LEN=512
class Glue(Dataset):
    def __init__(self,
                 task,
                 train=True,
                 num=0): # do bpe

        self.name = 'glue'
        self.train = train  
        self.task = task

        split = self.load_split() # loads split & labels
        # encoder only used for tokenization
        self.encoder = BertTokenizer.from_pretrained('google/multiberts-seed_0')
        self.dataset = split


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (encoded, len, tgt) where target is index of the target character class.
        """

        sentence = self.dataset[index]
        if self.task in ['cola', 'sst2']:
            tokens = self.encoder(sentence, return_tensors='pt')['input_ids'][0]
        elif self.task in ['mnli', 'mrpc', 'qnli', 'qqp', 'rte', 'stsb']:
            tokens = self.encoder(*sentence, return_tensors='pt')['input_ids'][0]
        # automatically truncating
        if len(tokens) > CONTEXT_LEN:
            return tokens[:CONTEXT_LEN], CONTEXT_LEN

        return tokens, len(tokens) # sequence of tokens, None label as placeholder

    def load_split(self):

        if self.train:
            dataset = load_dataset('glue', self.task)['train']
            if self.task in ['cola', 'sst2']:
                # skip first header line
                train_lines = [dataset[i]['sentence'] for i in range(len(dataset))]
            else:
                name0 = task_to_keys[self.task][0]
                name1 = task_to_keys[self.task][1]
                train_lines = [(dataset[i][name0], dataset[i][name1]) 
                               for i in range(len(dataset))]
        return train_lines

def pad_collate(batch, pad_tok):
    (xx, lens) = zip(*batch)
    xx_pad = pad_sequence(xx, batch_first=True, padding_value=pad_tok)
    return xx_pad, lens

def prepare_train_loaders(config):
    pad_tok = 0
    return {
        'full': torch.utils.data.DataLoader(
            Glue(train=True, 
                task=config['task']),
                batch_size=config['batch_size'],
                shuffle=config['shuffle_train'], 
                collate_fn=partial(pad_collate, pad_tok=pad_tok)
        )
    }

def prepare_test_loaders(config):
    pad_tok = 0
    return {
        'full': torch.utils.data.DataLoader(
            Glue(train=False, 
                task=config['task']),
                batch_size=config['batch_size'],
                shuffle=True, 
                collate_fn=partial(pad_collate, pad_tok=pad_tok)
        )
    }


if __name__ == "__main__":
    for task in ['cola', 'mrpc', 'qnli', 'qqp', 'rte', 'sst2', 'stsb']:
        config = {'batch_size': 4, 'num_workers': 4, 'shuffle_train': True, 'task': task}
        train_loader = prepare_train_loaders(config)
        x = next(iter(train_loader['full']))
        print(x)