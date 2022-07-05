# training config
batch_size = 8
n_step = 1000000
scheduler_step = 800000
save_step = 50000
log_checkpoint_step = 2000
lr = 1e-4
grad_clip = 1.
decay = .1
minf = .1
model_checkpoint = False
opt_checkpoint = False

# model config
base_scale = 1.
filter_size = 128
hyper_filter_size = 192
vbr = 0

# test config
gop_size = None
# betas = [1e-4, 2e-4, 4e-4, 8e-4, 1.6e-3, 3.2e-3, 6.4e-3]
# test_videos = ['BP', 'BT', 'HB', 'JK', 'RSG', 'SND', 'YR']
betas = [4e-4]
test_videos = ['RSG']
test_videos_path = '/home/ruihay1/playground/'
test_output_path = './test_npys/'

# data config
tbdir = 'tensorboard_logs'
data_config = {
    'dataset_name': 'vimeo',
    'data_path': '/srv/ssd0/bdl/ruihay1/dataset',
    'sequence_length': 3,
    'img_size': 256,
    'img_channel': 3
}
