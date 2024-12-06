method = 'FNO'
precision = '32'
# model
model_type = 'mlp'
in_channels = 2
out_channels = 2
hidden_channels = 256
n_modes = (2, 16, 16)
n_layers = 6
ndim = 3
# training
lr = 1e-3
batch_size = 32
sched = 'cosine'
opt = 'adamw'
warmup_epoch = 5