from csv import writer
import matplotlib.pyplot as plt
import pickle
import numpy as np


def list2csv(file, datalist):
    with open('{}.csv'.format(file), mode='a', newline='') as csv_file:
        writer_obj = writer(csv_file)
        writer_obj.writerow(datalist)

def list2newcsv(file, datalist):
    with open('{}.csv'.format(file), mode='w+') as csv_file:
        writer_obj = writer(csv_file)
        writer_obj.writerow(datalist)
    csv_file.close()

def list2newcsv_col(file, datalist):
    with open('{}.csv'.format(file), mode='w+', newline='') as csv_file:
        writer_obj = writer(csv_file)
        for val in datalist:
            writer_obj.writerow([val])
    csv_file.close()

def twodlist2csv(file, twodlist2csv):
    with open('{}.csv'.format(file), mode='a', newline='') as csv_file:
        writer_obj = writer(csv_file)
        for r in twodlist2csv:
            writer_obj.writerow(r)

def plot_reward(file, reward_trace):
    plt.figure(figsize=(15, 3))
    plt.plot(reward_trace)
    plt.savefig(file)
    plt.clf()

def load_bw_dict():
    return {
        "fccup": pickle.load(open("../bw/FCCUp10Train.pickle", 'rb')),
        "fcclow":pickle.load(open("../bw/FCCLow10Train.pickle", 'rb')),
        "lteup": pickle.load(open("../bw/LTEUp10Train.pickle", 'rb')),
        "ltelow":pickle.load(open("../bw/LTELow10Train.pickle", 'rb'))
    }

def load_bw_test():
    bw20 = []
    bw15 = []
    bw10 = []
    bw05 = []
    fccUpTest = pickle.load(open("../bw/FCCUp10test.pickle", 'rb'))
    fccUpTest = np.repeat(fccUpTest, 10, axis=1)
    fccLowTest = pickle.load(open("../bw/FCCLow10test.pickle", 'rb'))
    fccLowTest = np.repeat(fccLowTest, 10, axis=1)

    lteUpTest = pickle.load(open("../bw/LTEUp10test.pickle", 'rb'))
    lteLowTest = pickle.load(open("../bw/LTELow10test.pickle", 'rb'))

    for bw in np.concatenate([fccUpTest, lteUpTest, fccLowTest, lteLowTest]):
        if np.mean(bw) <= 2e6 and np.mean(bw) >= 1.5e6:
            bw20.append(bw)
        elif np.mean(bw) <= 1.5e6 and np.mean(bw) >= 1e6:
            bw15.append(bw)
        elif np.mean(bw) <= 1e6 and np.mean(bw) >= 0.5e6:
            bw10.append(bw)
        elif np.mean(bw) <= 0.5e6:
            bw05.append(bw)

    bw05 = np.concatenate([bw05, bw05, bw05, bw05])

    with open('../bw/bw20Test.pickle', 'wb') as handle:
        pickle.dump(bw20, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('../bw/bw15Test.pickle', 'wb') as handle:
        pickle.dump(bw15, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('../bw/bw10Test.pickle', 'wb') as handle:
        pickle.dump(bw10, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('../bw/bw05Test.pickle', 'wb') as handle:
        pickle.dump(bw05, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return bw20, bw15, bw10, bw05



if __name__ == '__main__':
    load_bw_test()
