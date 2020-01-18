import glob
import numpy
import model
import utils

def main():
    net = model.create()
    train(net)

def train(net):
    for epoch in range(100):
        for i, point, filename, label in utils.datapoints():
            utils.print_point(i, point, filename, label, epoch=epoch)
            net.fit(point,label, batch_size=1)

if __name__=="__main__":
    main()
