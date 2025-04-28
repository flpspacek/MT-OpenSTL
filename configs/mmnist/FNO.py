method = 'fno'
precision = '32'
# model
model_type = 'skip'
in_channels = 1
out_channels = 1
hidden_channels = 256
n_modes = (4, 16, 16)
n_layers = 4
ndim = 3
# training
lr = 5e-3
batch_size = 16
sched = 'cosine'
warmup_epoch = 5
