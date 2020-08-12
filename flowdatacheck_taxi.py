import numpy as np
import pickle
import json
import file_loader

import numpy as np
import pickle
import json

len_closeness = 4
len_period = 3
len_trend = 1

config_path="data.json"
config = json.load(open(config_path, "r"))
# how many timeslots per day (48 here)
timeslot_daynum = int(86400 / config["timeslot_sec"])
threshold = int(config["threshold"])
isFlowLoaded = False
isVolumeLoaded = False

def load_flow():
    flow_train = np.load(open(config["flow_train"], "rb"))["flow"] / config["flow_train_max"]
    flow_test = np.load(open(config["flow_test"], "rb"))["flow"] / config["flow_train_max"]
    isFlowLoaded = True
    return flow_train, flow_test

def load_volume():
    volume_train = np.load(open(config["volume_train"], "rb"))["volume"] / config["volume_train_max"]
    volume_test = np.load(open(config["volume_test"], "rb"))["volume"] / config["volume_train_max"]
    isVolumeLoaded = True
    return volume_train, volume_test

#a, b = load_flow()
train_data, test_data = load_volume()
train_data = np.transpose(train_data, (0, 3, 1, 2))
test_data = np.transpose(test_data, (0, 3, 1, 2))


def check_it(depends):
    for d in depends:
        if d < 0:
            return False
    return True


def create_dataset_train( T=48, len_closeness=len_closeness, len_trend=len_trend, TrendInterval=7, len_period=len_period, PeriodInterval=1):
    """current version
    """
    # offset_week = pd.DateOffset(days=7)
    #offset_frame = pd.DateOffset(minutes=24 * 60 // self.T)
    XC = []
    XP = []
    XT = []
    Y = []
    timestamps_Y = []
    depends = [range(1, len_closeness + 1), [PeriodInterval * T * j for j in range(1, len_period + 1)], [TrendInterval * T * j for j in range(1, len_trend + 1)]]

    i = max(T * TrendInterval * len_trend, T * PeriodInterval * len_period, len_closeness)
    while i < 1920:#训练集长度1920个实例
        Flag = True
        for depend in depends:
            if Flag is False:
                break
            Flag = check_it([i - j for j in depend])

        if Flag is False:
            i += 1
            continue
        x_c = [train_data[i - j] for j in depends[0]]
        x_p = [train_data[i - j] for j in depends[1]]
        x_t = [train_data[i - j] for j in depends[2]]
        y = train_data[i]
        if len_closeness > 0:
            XC.append(np.vstack(x_c))
        if len_period > 0:
            XP.append(np.vstack(x_p))
        if len_trend > 0:
            XT.append(np.vstack(x_t))
        Y.append(y)
        #timestamps_Y.append(timestamps[i])
        i += 1
    XC = np.asarray(XC)
    XP = np.asarray(XP)
    XT = np.asarray(XT)
    Y = np.asarray(Y)
    print("XC shape: ", XC.shape, "XP shape: ", XP.shape, "XT shape: ", XT.shape, "Y shape:", Y.shape)
    return XC, XP, XT, Y, timestamps_Y

def create_dataset_test( T=48, len_closeness=len_closeness, len_trend=len_trend, TrendInterval=7, len_period=len_period, PeriodInterval=1):
    """current version
    """
    # offset_week = pd.DateOffset(days=7)
    #offset_frame = pd.DateOffset(minutes=24 * 60 // self.T)
    XC = []
    XP = []
    XT = []
    Y = []
    timestamps_Y = []
    depends = [range(1, len_closeness + 1),
               [PeriodInterval * T * j for j in range(1, len_period + 1)],
               [TrendInterval * T * j for j in range(1, len_trend + 1)]]

    i = max(T * TrendInterval * len_trend, T * PeriodInterval * len_period, len_closeness)
    while i < 960:#训练集长度1920个实例
        Flag = True
        for depend in depends:
            if Flag is False:
                break
            Flag = check_it([i - j for j in depend])

        if Flag is False:
            i += 1
            continue
        x_c = [test_data[i - j] for j in depends[0]]
        x_p = [test_data[i - j] for j in depends[1]]
        x_t = [test_data[i - j] for j in depends[2]]
        y = test_data[i]
        if len_closeness > 0:
            XC.append(np.vstack(x_c))
        if len_period > 0:
            XP.append(np.vstack(x_p))
        if len_trend > 0:
            XT.append(np.vstack(x_t))
        Y.append(y)
        #timestamps_Y.append(timestamps[i])
        i += 1
    XC = np.asarray(XC)
    XP = np.asarray(XP)
    XT = np.asarray(XT)
    Y = np.asarray(Y)
    print("XC shape: ", XC.shape, "XP shape: ", XP.shape, "XT shape: ", XT.shape, "Y shape:", Y.shape)
    return XC, XP, XT, Y, timestamps_Y

train_XC, train_XP, train_XT, train_Y, train_timestamps_Y = create_dataset_train(len_closeness=len_closeness, len_period=len_period,
                                                             len_trend=len_trend)
test_XC, test_XP, test_XT, test_Y, test_timestamps_Y = create_dataset_test(len_closeness=len_closeness, len_period=len_period,
                                                            len_trend=len_trend)
XC_train = []
XP_train = []
XT_train = []
Y_train = []
XC_test = []
XP_test = []
XT_test = []
Y_test = []

XC_train.append(train_XC)
XP_train.append(train_XP)
XT_train.append(train_XT)
Y_train.append(train_Y)
XC_train = np.vstack(XC_train)
XP_train = np.vstack(XP_train)
XT_train = np.vstack(XT_train)
Y_train = np.vstack(Y_train)

XC_test.append(test_XC)
XP_test.append(test_XP)
XT_test.append(test_XT)
Y_test.append(test_Y)
XC_test = np.vstack(XC_test)
XP_test = np.vstack(XP_test)
XT_test = np.vstack(XT_test)
Y_test = np.vstack(Y_test)

X_train = []
X_test = []
for X_ in [XC_train, XP_train, XT_train]:
    X_train.append(X_)
for X_ in [XC_test, XP_test, XT_test]:
    X_test.append(X_)
#print('finish')


