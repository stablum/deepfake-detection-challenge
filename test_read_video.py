import glob
import skvideo.io
import numpy
import humanize
import utils
def main():
    for i, slices, filename, label in utils.datapoints():
        for curr_slice in slices:
            utils.print_point(i, curr_slice, filename,label)

if __name__=="__main__":
    main()

