import numpy as np
import time
import utils

def predictions(net):
    ret = []
    for i, slices, filename, label in utils.datapoints():
        for curr_slice in slices:
            start = time.time()
            cls = net.predict(curr_slice, batch_size=1)
            end = time.time()
            print("prediction:",cls, "label:",label)
            print("elapsed time to predict:",end - start)
            ret.append((cls[0][0],label[0][0]))
    return ret

def avg_error(net,predictions):
    diffs = []
    for cls, label in predictions:
        diff = np.abs(cls - label)
        diffs.append(diff)
    diff_a = np.array(diffs)
    rounded = np.round(diff_a) # quantize 0/1
    quantized = np.mean(rounded)
    unquantized = np.mean(diff_a)
    return quantized, unquantized

def class_corrects(net, predictions):
    totals = {0:0.0, 1:0.0} # dataset amounts per class; key: class
    correct = {0:0.0, 1:0.0} # correct counts per class; key: class
    for cls, label in predictions:
        totals[label] += 1.0
        if int(np.abs(cls)) == int(label):
            correct[label] += 1.0
    ret = {}
    for label in totals.keys():
        ret[label] = correct[label]/totals[label]
    print("totals",totals,"correct",correct,"ret",ret)
    return ret

