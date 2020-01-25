import glob
import numpy
import model
import utils
import numpy as np
import mlflow
import time
import metrics
import config

def main():
    utils.log_config()
    net = model.create()
    utils.log_net(net)
    train(net)

def train(net):
    for epoch in range(config.n_epochs):
        epoch_start = time.time()
        mlflow.log_metric("epoch",epoch)
        predictions = metrics.predictions(net)
        qu, uq = metrics.avg_error(net, predictions)
        print("epoch:",epoch,"avg_error - quantized:",qu,"unquantized:",uq)
        mlflow.log_metric("unquantized_avg_error",uq)
        mlflow.log_metric("quantized_avg_error",qu)
        class_corrects = metrics.class_corrects(net, predictions)
        for cls,corrects in class_corrects.items():
            mlflow.log_metric('class_corrects_'+str(int(cls)),corrects)

        fit_times = []
        losses = []
        accuracies = []
        for i, slices, filename, label in utils.datapoints():
            utils.print_point(i, slices[0], filename, label, epoch=epoch)
            for curr_slice in slices:
                start = time.time()
                history = net.fit(curr_slice,label, batch_size=1)
                losses.append(history.history['loss'][0])
                accuracies.append(history.history['accuracy'][0])
                end = time.time()
                fit_time = end-start
                print("elapsed time to fit:",fit_time)
                fit_times.append(fit_time)
        epoch_end = time.time()
        mlflow.log_metric("avg_datapoint_fit_time",np.avg(fit_times))
        mlflow.log_metric("epoch_mins",(epoch_end-epoch_start)/60.0)
        mlflow.log_metric("avg_loss",np.avg(losses))
        mlflow.log_metric("avg_accuracy",np.avg(accuracies))


if __name__=="__main__":
    main()

