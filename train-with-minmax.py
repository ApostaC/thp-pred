from __future__ import print_function
import torch.nn.functional as F
from torch.autograd import Variable
import sklearn.preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import torch
import time
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

### GLOBAL CONFIGURATION
config = {
    'PREFIX'        : 'smoothed',   # the unique identifier for result dir and ckpt file name
    'CKPT_EPOCH'    : 15,           # do check pointing every 'CKPT_EPOCH' epoches 
    'BATCHSIZE'     : 75,           # the batchsize for training
    'INPUTLEN'      : 100,          # the input length (100 means 100 seconds for input)
    'OUTPUTLEN'     : 5,            # the output length (5 means 5 seconds for prediction
    'LEARNING_RATE' : 0.001,        # learning rate
    'EPOCH'         : 1000          # stop at how many epoches
}
def validate_config():
    config['PREFIX'] = str(config['PREFIX'])
    config['CKPT_EPOCH'] = int(config['CKPT_EPOCH'])
    config['BATCHSIZE'] = int(config['BATCHSIZE'])
    config['INPUTLEN'] = int(config['INPUTLEN'])
    config['OUTPUTLEN'] = int(config['OUTPUTLEN'])
    config['LEARNING_RATE'] = float(config['LEARNING_RATE'])
    config['EPOCH'] = int(config['EPOCH'])


PREFIX          = None
CKPT_EPOCH      = None
FEATURE_CNT     = 6         # thp, avg-signal, delta-signal, have-handover, rtt, loss

def smooth_row(mat, rowid, kernel):
    row = mat[rowid]
    row = np.convolve(row, kernel, mode = 'same')
    mat[rowid] = row
    return mat

####################
# read datafile and parse it as input matrix
####################
thpfiles = []
with open("../newdata/final/dir.txt", "r") as fdir:
    for lines in fdir:
        thpfiles.append(lines.rstrip('\n'))
#thpfiles = thpfiles[:12]

print("Loading data...")
mats = []
for fname in thpfiles:
    mat = []
    with open(fname, "r") as fin:
        print("Reading", fname)
        for line in fin:
            val = np.zeros(FEATURE_CNT)
            splited_line = line.rstrip('\n').split(' ')
            for ind, num in zip(np.arange(len(splited_line)), splited_line):
                val[ind] = float(num)
            mat.append(val)
        mat = np.array(mat) 
    # modify the thp 
    t_mat = mat.transpose().copy()
    mat = smooth_row(t_mat, 0, [0.2, 0.2, 0.2, 0.2, 0.2]).transpose().copy()
    mats.append(mat)

# use data 1 as final test
final_mat = mats[0]
mats = mats[1:]
mats = np.array(mats)

####################
# Data Slicer
####################
class DataSlicer:
    def __init__(self, batch_size, input_len, output_len):
        self.flag = False   # if flag, output is availiable
        self.BATCHSIZE = batch_size
        self.INPUTLEN = input_len
        self.OUTPUTLEN = output_len

    def _build_timeseries(self, mat, o_col_ind):
        """
        mat: [timelen, features]
        o_col_ind: which feature the output is
        returns i_series and o_series
        """
        dim0 = len(mat) - self.INPUTLEN - self.OUTPUTLEN
        if dim0 < 0:
            print("DataSlicer::_build_timeseries: input length is too short")
            raise Exception("DataSlicer::_build_timeseries: input length is too short")
        dim1 = len(mat[0])

        x = np.zeros((dim0, self.INPUTLEN, dim1), 'float64')
        y = np.zeros((dim0, self.OUTPUTLEN), 'float64')

        for i in range(dim0):
            x[i] = mat[i : self.INPUTLEN+i]
            y[i] = mat[i+self.INPUTLEN : i+self.INPUTLEN+self.OUTPUTLEN, o_col_ind]
        return x, y

    def _batching(self, arr):
        """
        input is numpy array
        trims arr to a size that's divisible by BATCH_SIZE
        and slice them in batch
        returns batch[number of batch], size each item in batch array is batchsize
        output is list of numpy array
        """
        no_of_rows_drop = len(arr) % self.BATCHSIZE
        num_of_batches = len(arr) // self.BATCHSIZE 
        if(no_of_rows_drop > 0):
            arr = arr[:-no_of_rows_drop]
        batches = []
        for i in range(0, len(arr), self.BATCHSIZE):
            batches.append(arr[i:i+self.BATCHSIZE])
        return batches

    def randomize(self, idata, odata):
        """
        idata: len * selfshape
        odata: len * selfshape
        """
        buf = []
        ind = 0
        for i, o in zip(idata, odata):
            buf.append((i,o))
            ind += 1
        buf = np.array(buf)
        np.random.shuffle(buf)
        reti = np.zeros(idata.shape)
        reto = np.zeros(odata.shape)
        ind = 0
        for i, o in buf:
            reti[ind] = i
            reto[ind] = o
            ind += 1
        return reti, reto

        

    def slice(self, mats):
        """
        input data of multiple days
        output the batched i_train, o_train, i_test, o_test
        """
        ri_train = []
        ro_train = []
        ri_test = []
        ro_test = []
        min_max_scaler = MinMaxScaler()
        for mat in mats:
            # df_train, df_test = train_test_split(mat, train_size=0.8, test_size=0.2, shuffle=False)
            # Normalization
            x_total = min_max_scaler.fit_transform(mat)
            #x_train = min_max_scaler.fit_transform(df_train)
            #x_test = min_max_scaler.fit_transform(df_test)
            #x_train = sklearn.preprocessing.scale(df_train)
            #x_test = sklearn.preprocessing.scale(df_test)
            i_total, o_total = self._build_timeseries(x_total, 0)
            i_total, o_total = self.randomize(i_total, o_total)
            i_train, i_test = train_test_split(i_total, train_size = 0.8, shuffle = False)
            o_train, o_test = train_test_split(o_total, train_size = 0.8, shuffle = False)
            #i_train, o_train = self._build_timeseries(x_train, 0)
            #i_test, o_test = self._build_timeseries(x_test, 0)
            ri_train.extend(self._batching(i_train))
            ro_train.extend(self._batching(o_train))
            ri_test.extend(self._batching(i_test))
            ro_test.extend(self._batching(o_test))
        print("Input and Output length is", self.INPUTLEN, self.OUTPUTLEN)
        print("Train and Test Size in Batch", len(ri_train), len(ro_train), len(ri_test), len(ro_test))
        print("Batch size is ", self.BATCHSIZE)
        return ri_train, ro_train, ri_test, ro_test
            

class Network(nn.Module):
    #def __init__(self, indim, timestep, outdim, outlen, batchsize, device):
    def __init__(self, indim, timestep, outdim, outlen, batchsize):
        super(Network, self).__init__()
        self.indim = indim
        self.outdim = outdim
        self.timestep = timestep
        self.outlen = outlen
        self.hidden_dim = 80
        self.lstm = nn.LSTM(indim, self.hidden_dim, batch_first=True)
        self.drop = nn.Dropout(p = 0.2)
        self.dense1 = nn.Linear(self.hidden_dim, 20) # magic number 20 here
        self.dense2 = nn.Linear(20, 1)
        #self.dense2 = nn.Linear(self.hidden_dim, 1)
        self.batchsize = batchsize
        self.hidden = self.init_hidden(batchsize)

    def setbatchsize(self, batchsize):
        self.batchsize = batchsize

    def init_hidden(self, batchsize):
        return (Variable(torch.zeros(1, batchsize, self.hidden_dim, dtype = torch.double).cuda()),
                Variable(torch.zeros(1, batchsize, self.hidden_dim, dtype = torch.double).cuda()))
        # if not running on GPU please use following lines
        #return (Variable(torch.zeros(1, batchsize, self.hidden_dim, dtype = torch.double)),
        #    Variable(torch.zeros(1, batchsize, self.hidden_dim, dtype = torch.double)))


    def forward(self, x):
        # x : batch * timestep * features
        self.hidden[0].detach_()
        self.hidden[1].detach_()

        x, self.hidden = self.lstm(x, self.hidden)
        x = self.drop(x)
        x = torch.tanh(self.dense1(x))
        x = torch.sigmoid(self.dense2(x[:,-self.outlen:,:]))
        return x

class Trainer:
    def __init__(self, batch_size, input_len, output_len, lr, epoch):
        self.BATCHSIZE = batch_size
        self.INPUTLEN = input_len
        self.OUTPUTLEN = output_len
        self.LR = lr
        self.EPOCH = epoch
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.network = Network(indim = FEATURE_CNT,
                timestep = self.INPUTLEN, 
                outdim = 1,
                outlen = self.OUTPUTLEN,
                batchsize = self.BATCHSIZE
                )
        self.network.to(self.device)

    def train(self, rawmats):
        # prepare data
        slicer = DataSlicer(self.BATCHSIZE, self.INPUTLEN, self.OUTPUTLEN)
        itr, otr, ite, ote = slicer.slice(rawmats) # nbatches * batchsz * timestep * features

        # prepare network
        network = self.network
        network.double()
        optimizer = optim.RMSprop(network.parameters(), lr = self.LR)#, weight_decay = 0.001)
        criterion = nn.MSELoss()

        batch = torch.FloatTensor(itr[0])
        print(batch.shape)
        # train
        train_loss = []
        test_loss = []
        fout = open("results-{}/running.csv".format(PREFIX), "w")
        for epoch in range(self.EPOCH + 1):
            trls = 0.
            lasttime = time.time()
            for batch in range(len(itr)):
                _ = []
                input = torch.from_numpy(itr[batch]).to(self.device)
                target = torch.from_numpy(otr[batch].reshape((self.BATCHSIZE, self.OUTPUTLEN, 1))).to(self.device)
                #def closure():
                optimizer.zero_grad()
                out = network(input)
                loss = criterion(out, target)
                a = list(network.parameters())[0].clone()
                loss.backward()
                _.append(loss.item())
                #return loss
                optimizer.step()
                b = list(network.parameters())[0].clone()
                #print(list(network.parameters())[0].grad)
                #print(torch.equal(a, b))
                if torch.equal(a, b):
                    print("Warning! Parameter doesn't change!")
                trls += _[0]
                if batch % 10 == 1:
                    print("Epoch {}, batch {}, loss {}, time {}".format(epoch, batch, trls / batch, time.time() - lasttime))
                    lasttime = time.time()
            trls /= len(itr)

            # Test per epoch
            tels = 0.
            with torch.no_grad():
                test_input = torch.from_numpy(ite[0]).to(self.device)
                test_target = torch.from_numpy(ote[0].reshape((self.BATCHSIZE, self.OUTPUTLEN, 1))).to(self.device)
                pred = network(test_input)
                loss = criterion(pred, test_target)
                tels = loss.item()

            test_loss.append(tels)
            train_loss.append(trls)
            outline = "{} {:.6f} {:.6f}\n".format(epoch, trls, tels)
            fout.write(outline)
            fout.flush()

            print("=======================================")
            print("Train loss {}, test loss {}".format(trls, tels))
            print("=======================================")

            if epoch > 0 and epoch % CKPT_EPOCH == 0:
                torch.save(network,'models/test-{}.pt'.format(epoch))
                tester = Tester(self.network, input_len = self.INPUTLEN, device = self.device)
                tester.test(final_mat,batchsize = self.BATCHSIZE, outputlen = 2)

        fout.close()
        with open("results-{}/train-loss.csv".format(PREFIX),"w") as fout:
            for i in range(self.EPOCH):
                outline = "{} {:.6f} {:.6f}\n".format(i, train_loss[i], test_loss[i])
                print(outline)
                fout.write(outline)
                fout.flush()

class Tester:
    def __init__(self, network, input_len, device = None):
        self.network = network
        self.INPUTLEN = input_len
        self.device = device

    def test(self, mat, batchsize, outputlen):
        # mat: totaltime * features
        # network input: batchsize * input_len * features
        sca = MinMaxScaler()
        alist = []
        tlist = [] # output should start from where
        mat = sca.fit_transform(mat)
        print(len(mat))
        result = np.zeros(len(mat))
        for i in range(0,len(mat),outputlen):
            tempmat = mat[i:i+self.INPUTLEN]
            if len(tempmat) < self.INPUTLEN:
                break
            alist.append(tempmat)
            tlist.append(i + self.INPUTLEN)
        drop = len(alist) % batchsize
        if drop > 0:
            alist = alist[:-drop]
        print(len(alist))
        # alist: totalblocks * input_len * features
        for i in range(0,len(alist),batchsize):
            input = torch.DoubleTensor(alist[i:i+batchsize]).to(self.device)
            out = self.network(input)
            out = out.detach().cpu().numpy()    # out: totalblocks * outputlen
            for j in range(len(out)):   # the id of the batch: i + j
                t = i + j
                t = tlist[t]
                for k in range(outputlen):
                    result[t + k] = out[j][k]
        with open('results-{}/test.txt'.format(PREFIX), "w") as fout:
            for i in range(len(mat)):
                outline = "{} {} {}\n".format(i, mat[i][0], result[i])
                fout.write(outline)
            fout.flush()

        return

    
#BATCHSIZE = 75
#INPUTLEN = 100
#OUTPUTLEN = 5
#LEARNING_RATE = 0.001
#EPOCH = 1000

import sys

def ReadParams(filename):
    with open(filename, 'r') as fin:
        for line in fin:
            line = [_.strip(' \t') for _ in line.rstrip('\n').split('=')]
            k, v = line[:2]
            if k in config:
                config[k] = v
                print("[ INFO ] Reset configuration item: {} to {}".format(k, v))
    validate_config()


# Usage python3 <this-file>.py <load|train> <filename>
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 {} <load | train> <filename>".format(sys.argv[0]))
        exit(0)

    if sys.argv[1] == 'train':
        ReadParams(sys.argv[2])

    PREFIX = config['PREFIX']
    CKPT_EPOCH = config['CKPT_EPOCH']
    trainer = Trainer(batch_size = config['BATCHSIZE'], input_len = config['INPUTLEN'], output_len = config['OUTPUTLEN'],
                    lr = config['LEARNING_RATE'], epoch = config['EPOCH'])

    if sys.argv[1] == 'load':
        print("Use {} to build the network".format(sys.argv[2]))
        net = torch.load(sys.argv[2])
        net.eval()
    elif sys.argv[1] == 'train':
        print("Training Routine")
        #trainer = Trainer(batch_size = BATCHSIZE, input_len = INPUTLEN, output_len = OUTPUTLEN,
        #            lr = LEARNING_RATE, epoch = EPOCH)
        trainer.train(mats)
        net = trainer.network

    tester = Tester(net, input_len = config['INPUTLEN'], device =trainer.device)
    tester.test(final_mat,batchsize = config['BATCHSIZE'], outputlen = 2)

