import os
from data import srdata

class DIV2K(srdata.SRData):
    def __init__(self, args, name='ITW2K', train=True, benchmark=False):
        data_range = [r.split('-') for r in args.data_range.split('/')]
        if train:
            data_range = data_range[0]
        else:
            if args.test_only and len(data_range) == 1:
                data_range = data_range[0]
            else:
                data_range = data_range[1]

        self.begin, self.end = list(map(lambda x: int(x), data_range))
        super(DIV2K, self).__init__(
            args, name=name, train=train, benchmark=benchmark
        )

    def _scan(self):
        names_hr, names_lr = super(DIV2K, self)._scan()
        names_hr = names_hr[self.begin - 1:self.end]
        names_lr = [n[self.begin - 1:self.end] for n in names_lr]

        return names_hr, names_lr

    def _set_filesystem(self, dir_data):
        super(DIV2K, self)._set_filesystem(dir_data)
        #print(self.apath)
        #find the dataset
        # self.apath = '/home/littlepure/dataset/IFrame/NEW1080p/'
        # self.dir_hr = os.path.join(self.apath, 'DIV2K_train_HR_SR_1')
        #self.dir_lr = os.path.join(self.apath, 'DIV2K_train_LR_bicubic_SR_1')
        #self.apath = '/home/ljm/datasets/SR_ICCV/4k/lon_15s_1/'
        #self.apath = '/home/ljm/datasets/SR_ICCV/3/15s/3_15s_1/'
        #self.apath = '/home/ljm/datasets/SR_ICCV/lol/lol_1min_1'
        #self.apath = '/home/ljm/datasets/REDS/020'
        self.dir_hr = os.path.join(self.apath, 'DIV2K_train_HR')
        self.dir_lr = os.path.join(self.apath, 'DIV2K_train_LR_bicubic')
        #self.dir_hr = os.path.join(self.apath, '/train/ori/')
        #self.dir_lr = os.path.join(self.apath, '/train/2x/')
        if self.input_large: self.dir_lr += 'L'

