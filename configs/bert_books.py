config = {
'dataset': [
        {
            'name': 'books',
            'shuffle_train': False,
            'batch_size': 1,
            'train_fraction': 0.001,
            'sorted': False,
            'tokenizer': 'bert',
            'num':0
        }
    ],
    'model': {
        'name': 'bert',
        'dir': 'google',
        'bases': []
    },
     'model_names': {
        'model1':'multiberts-seed_0',
        'model2':'multiberts-seed_1'
    },
    'parallel_data': False,
    'merging_fn': 'match_tensors_permute',
    'merging_metrics': ['covariance'],
}


