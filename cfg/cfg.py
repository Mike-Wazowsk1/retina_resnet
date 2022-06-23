class Cfg:
    def __init__(self):
        self.retina_cfg = {
            'name': 'mobilenet0.25',
            'min_sizes': [[16, 32], [64, 128], [256, 512]],
            'steps': [8, 16, 32],
            'variance': [0.1, 0.2],
            'clip': False,
            'loc_weight': 2.0,
            'gpu_train': False,
            'batch_size': 32,
            'ngpu': 1,
            'epoch': 250,
            'decay1': 190,
            'decay2': 220,
            'image_size': 640,
            'pretrain': True,
            'return_layers': {'stage1': 1, 'stage2': 2, 'stage3': 3},
            'in_channel': 32,
            'out_channel': 64
        }
        self.retina_path = 'Retina_state.pt'
        self.resnet_path = 'best_checkpoint.tar'
        self.device = 'cpu'
        self.path_to_img = '/face'
