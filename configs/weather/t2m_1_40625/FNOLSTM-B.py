method = 'fnolstm'
precision = '32'
# model args
in_T = 4
in_channels = 2
num_hidden = 32
fno_block_args = {
    'model_type': 'skip',
    'in_channels': num_hidden,
    'out_channels': num_hidden,
    'hidden_channels': 64,
    'n_modes': (16, 16),
    'n_layers': 4,
    'ndim': 2,
}
# training
lr = 5e-3
batch_size = 32
sched = 'cosine'
opt = 'adamw'
warmup_epoch = 5