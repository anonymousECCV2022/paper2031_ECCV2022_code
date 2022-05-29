import torch
import numpy as np
import random
import torch.backends.cudnn as cudnn

import utility
import data
import model
import loss
from option import args
from trainer_visual import Trainer_visual
from trainer_sidetuning import Trainer_sidetuning
from trainer_org import Trainer_org
from trainer_tcloss import Trainer_tcloss
import pdb
torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)

def main():
    global model
    if args.data_test == ['video']:
        from videotester import VideoTester
        model = model.Model(args, checkpoint)
        t = VideoTester(args, model, checkpoint)
        t.test()
    else:
        if checkpoint.ok:
            #pdb.set_trace()
            loader = data.Data(args)
            _model = model.Model(args, checkpoint)
            args.pre_train = "/home/ljm/EDSR-TCloss/experiment/edsr_p96_3_15s_1_seg1-3_0114_adafm_S2/model/model_best.pt"
            checkpoint1 = utility.checkpoint(args)
            _model1 = model.Model(args, checkpoint1)
            t = Trainer_visual(args, loader, _model,_model1)
            while not t.terminate():
                t.train()
                t.test()

            checkpoint.done()

if __name__ == '__main__':
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    cudnn.benchmark = False
    cudnn.deterministic = True
    main()
