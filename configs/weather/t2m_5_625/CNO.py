method = 'cno'
model_type = 'CNO'
precision = 'bf16-mixed'
# model
dim = 3
in_dim = 2
out_dim = 2
size = (4, 32, 32)
N_layers = 4
N_res = 4
N_res_neck = 4
channel_multiplier = 16
use_bn = False
# training
lr = 1e-3
batch_size = 32
sched = 'cosine'
warmup_epoch = 5
