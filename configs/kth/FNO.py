method = 'FNO'
precision = '32'
# model
model_type = 'skip'
in_channels = 1
out_channels = 1
hidden_channels = 256
n_modes = (8, 64, 64)
n_layers = 4
ndim = 3
# training
lr = 1e-3
batch_size = 32
sched = 'cosine'
opt = 'adamw'
warmup_epoch = 5
