import numpy as np

kernel = [0.2, 0.2, 0.2, 0.2, 0.2]
data = []
with open('tmp.dat', 'r') as fin:
    for line in fin:
        data.append(float(line.rstrip('\n')))

res = np.convolve(data, kernel, mode='same')
with open('plot.dat', 'w') as fout:
    cnt = 0
    for a, b in zip(data, res):
        fout.write("{} {} {}\n".format(cnt, a, b))
        cnt += 1
