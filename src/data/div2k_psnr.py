import os
import random
import pickle
from data import srdata
from data import common
import imageio
from option import args

class DIV2K_PSNR(srdata.SRData):
    def __init__(self, args, name='DIV2K_PSNR', train=True, benchmark=False):
        data_range = [r.split('-') for r in args.data_range.split('/')]
        if train:
            data_range = data_range[0]
        else:
            if args.test_only and len(data_range) == 1:
                data_range = data_range[0]
            else:
                data_range = data_range[1]

        self.begin, self.end = list(map(lambda x: int(x), data_range))
        self.data_partion = args.data_partion
        super(DIV2K_PSNR, self).__init__(
            args, name='DIV2K', train=train, benchmark=benchmark
        )

    def _scan(self):
        names_hr, names_lr = super(DIV2K_PSNR, self)._scan()
        names_hr = names_hr[self.begin - 1:self.end]
        names_lr = [n[self.begin - 1:self.end] for n in names_lr]

        return names_hr, names_lr

    def _set_filesystem(self, dir_data):
        super(DIV2K_PSNR, self)._set_filesystem(dir_data)
        self.apath = dir_data
        #self.apath = '/home/littlepure/dataset/IFrame/NEW1080p/'
        self.dir_hr = os.path.join(self.apath, 'DIV2K_train_HR')
        self.dir_lr = os.path.join(self.apath, 'DIV2K_train_LR_bicubic')
        if self.input_large: self.dir_lr += 'L'

    def __getitem__(self, idx):
        lr, hr, filename, psnr_index = self._load_file(idx)
        
        pair = self.get_patch(lr, hr, psnr_index)
        pair = common.set_channel(*pair, n_channels=self.args.n_colors)
        pair_t = common.np2Tensor(*pair, rgb_range=self.args.rgb_range)

        return pair_t[0], pair_t[1], filename

    def _load_file(self, idx):
        idx = self._get_index(idx)
        f_hr = self.images_hr[idx]
        f_lr = self.images_lr[self.idx_scale][idx]
        #org

        # f_psnr = f_lr.replace('x3.pt','_psnr_map.pt')
        # f_psnr = f_psnr.replace('bin/DIV2K_train_LR_bicubic/X3','psnr_map')
        cur_scale = str(args.scale[0])
        use_scale = 'x'+cur_scale
        f_psnr = f_lr.replace(use_scale+'.pt','_psnr_map.pt')
        f_psnr = f_psnr.replace('bin/DIV2K_train_LR_bicubic/X'+cur_scale,'psnr_map')
        #f_psnr = os.path.join(self.dir_data,"psnr")
       
        #new
        # if idx<=55:
        #     f_lr1 = self.images_lr[self.idx_scale][1]
        #     f_psnr = f_lr1.replace('x4.pt','_psnr.pt')
        # else:
        #     f_lr1 = self.images_lr[self.idx_scale][56]
        #     f_psnr = f_lr1.replace('x4.pt','_psnr.pt')

        filename, _ = os.path.splitext(os.path.basename(f_hr))
        if self.args.ext == 'img' or self.benchmark:
            hr = imageio.imread(f_hr)
            lr = imageio.imread(f_lr)
        elif self.args.ext.find('sep') >= 0:
            with open(f_hr, 'rb') as _f:
                hr = pickle.load(_f)
            with open(f_lr, 'rb') as _f:
                lr = pickle.load(_f)
            with open(f_psnr, 'rb') as _f:
                psnr_index = pickle.load(_f)
        n_patch = int(psnr_index.shape[0]*self.data_partion)
        # with open("../../pivot.txt", "a") as file_object:  
        #     print('1111')
        #     file_object.write(self.apath)
        #     file_object.write(" ")
        #     file_object.write(str(self.data_partion))
        #     file_object.write(": ")
        #     file_object.write(str(psnr_index[n_patch]))
        #     file_object.write("\n ")
        #     # file_object.write("lxq")
        # assert(0)
        return lr, hr, filename, psnr_index

    def get_patch(self, lr, hr, psnr_index):

        scale = self.scale[self.idx_scale]
        if self.train:
            
            lr, hr = common.get_patch(
                lr, hr, 
                patch_size=self.args.patch_size,
                scale=scale,
                multi=(len(self.scale) > 1),
                input_large=self.input_large,
                psnr_index=psnr_index,
                data_partion=self.data_partion,            
            )
            if not self.args.no_augment: lr, hr = common.augment(lr, hr)
        else:
            ih, iw = lr.shape[:2]
            hr = hr[0:ih * scale, 0:iw * scale]
        

        return lr, hr