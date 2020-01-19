import glob
import numpy
import model
import utils
import numpy as np
import mlflow
import metrics

def main():
    utils.log_config()
    net = model.create()
    utils.log_net(net)
    train(net)

def train(net):
    for epoch in range(100):
        mlflow.log_metric("epoch",epoch)
        predictions = metrics.predictions(net)
        qu, uq = metrics.avg_error(net, predictions)
        print("epoch:",epoch,"avg_error - quantized:",qu,"unquantized:",uq)
        mlflow.log_metric("unquantized_avg_error",uq)
        mlflow.log_metric("quantized_avg_error",qu)
        class_corrects = metric.class_corrects(net, predictions)
        for cls,corrects in class_corrects.items():
            mlflow.log_metric('class_corrects_'+str(int(cls)),corrects)

        for i, point, filename, label in utils.datapoints():
            utils.print_point(i, point, filename, label, epoch=epoch)
            net.fit(point,label, batch_size=1)
            classes = net.predict(point, batch_size=1)
            print("prediction:",classes, "label:",label)


if __name__=="__main__":
    main()
