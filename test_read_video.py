import glob
import skvideo.io
import numpy
import humanize
import utils
def main():
    for i, point , filename, label in utils.datapoints():
        utils.print_point(i, point, filename,label)

if __name__=="__main__":
    main()

