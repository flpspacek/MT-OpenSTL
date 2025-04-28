method = 'cno'
model_type = 'CNO'
precision = 'bf16-mixed'
# model
dim = 3
in_dim = 2
out_dim = 2
size = (4, 32, 32)
N_layers = 3
N_res = 6
N_res_neck = 6
channel_multiplier = 64
use_bn = False
# training
lr = 2e-4
batch_size = 32
sched = 'cosine'
opt = 'adamw'
warmup_epoch = 5
