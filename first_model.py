import glob
import numpy
import model
import utils
import numpy as np
import mlflow

def main():
    utils.log_config()
    net = model.create()
    utils.log_net(net)
    train(net)

def avg_error(net):
    diffs = []
    for i, point, filename, label in utils.datapoints():
        cls = net.predict(point, batch_size=1)
        diff = np.abs(cls[0][0] - label[0][0])
        diffs.append(diff)
    diff_a = np.array(diffs)
    rounded = np.round(diff_a) # quantize 0/1
    quantized = np.mean(rounded)
    unquantized = np.mean(diff_a)
    return quantized, unquantized

def train(net):
    for epoch in range(100):
        mlflow.log_metric("epoch",epoch)
        qu, uq = avg_error(net)
        print("epoch:",epoch,"avg_error - quantized:",qu,"unquantized:",uq)
        mlflow.log_metric("unquantized_avg_error",uq)
        mlflow.log_metric("quantized_avg_error",qu)
        for i, point, filename, label in utils.datapoints():
            utils.print_point(i, point, filename, label, epoch=epoch)
            net.fit(point,label, batch_size=1)
            classes = net.predict(point, batch_size=1)
            print("prediction:",classes, "label:",label)

if __name__=="__main__":
    main()

