from .books import Books
from .glue import Glue

books = {
    'wrapper': Books,
    'batch_size': 8,
    'type': 'books',
    'num_workers': 8,
    'shuffle_train': False,
    'shuffle_test': False,
    'sorted': False,
}


glue = {
    'wrapper': Glue,
    'batch_size': 8,
    'type': 'glue',
    'num_workers': 8,
    'shuffle_train': False,
    'shuffle_test': False,
    'task': 'mnli'
}
