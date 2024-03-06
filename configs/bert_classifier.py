config = {
    'dataset': [
        {
            'name': 'glue',
            'batch_size': 1,
            'train_fraction': 0.001,
            'shuffle_train': False,
        }
    ],
    'model': {
        'name': 'bert',
        'dir': 'trained/multiberts/',
        'bases': []
    },
     'model_names': {
        'model1':'mnli/seed_0',
        'model2':'mnli/seed_1'
    },
    'parallel_data': False,
    'merging_fn': 'match_tensors_permute',
    'merging_metrics': ['covariance'],
}

