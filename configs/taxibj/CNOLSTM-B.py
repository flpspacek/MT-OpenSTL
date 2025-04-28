method = 'cnolstm'
model_type = 'CNOLSTM-B'
precision = 'bf16-mixed'
# model
in_T = 4
in_channels = 2
num_hidden = 32
n_layers = 1
cno_block_args = {
'dim': 2,
'in_dim': num_hidden,
'out_dim': num_hidden,
'size': (32, 32),
'N_layers': 3,
'N_res': 4,
'N_res_neck': 4,
'channel_multiplier': 128,
'use_bn': False,
}
# training
lr = 2e-4
batch_size = 32
sched = 'cosine'
opt = 'adamw'
warmup_epoch = 5