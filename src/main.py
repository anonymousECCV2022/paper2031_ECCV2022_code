import torch
import numpy as np
import random
import torch.backends.cudnn as cudnn

import utility
import data
import model
import loss
from option import args

from trainer_maml import Trainer_maml
from trainer_gradmask import Trainer_gradmask
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
            print(_model.state_dict().keys())
            _loss = loss.Loss(args, checkpoint) if not args.test_only else None
            
            if args.find_mask:
                t = Trainer_gradmask(args, loader, _model, _loss, checkpoint)
            elif args.maml:
                t = Trainer_maml(args, loader, _model, _loss, checkpoint)
            
            if args.patchnet_test:
                t.test_pn()
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
