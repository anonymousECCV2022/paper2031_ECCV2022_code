from torch.utils.data.sampler import Sampler

class MySampler(Sampler):

    def __init__(self, data_source, batchsize, seg_split=120):
        self.data_source = data_source
        self.seg1_list = [i for i in range(seg_split)]
        self.seg2_list = [i for i in range(seg_split, len(data_source))]
        self.list = []
        while len(self.seg1_list)>=batchsize and len(self.seg2_list)>=batchsize:
            for i in range(batchsize):
                self.list.append(self.seg1_list.pop())
            for i in range(batchsize):
                self.list.append(self.seg2_list.pop())
        if len(self.seg1_list)<batchsize:
            for i in self.seg1_list:
                self.list.append(i)
            for i in self.seg2_list:
                self.list.append(i)            
        else:
            for i in range(batchsize):
                self.list.append(self.seg1_list.pop())   
            for i in self.seg2_list:
                self.list.append(i)  
            for i in self.seg1_list:
                self.list.append(i)                    

    def __iter__(self):
        return iter(self.list)

    def __len__(self):
        return len(self.data_source)

S = MySampler([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],4,seg_split=9)
print([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
print(S.list)
print(next(iter(S)))
