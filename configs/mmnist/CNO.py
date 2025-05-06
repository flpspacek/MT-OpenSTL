method = 'cno'
model_type = 'CNO'
precision = 'bf16-mixed'
# model
dim = 3
in_dim = 1
out_dim = 1
size = (10, 64, 64)
N_layers = 3
N_res = 4
N_res_neck = 4
channel_multiplier = 64
use_bn = False
# training
lr = 2e-4
batch_size = 32
sched = 'cosine'
warmup_epoch = 5
