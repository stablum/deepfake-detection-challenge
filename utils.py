import skvideo.io
import humanize
import glob
import numpy as np
import json
import os
import random

def read_metadata():
    f = open('train_sample_videos/metadata.json')
    content = f.read()
    ret = json.loads(content)
    return ret

def print_point(i, point, filename,label, **kwargs):
    size = humanize.naturalsize(point.nbytes)

    print('index', i, 'shape', point.shape, filename, "uncompressed size",size,"label", label, kwargs)

def slice_videodata(videodata):
    orig_shape = videodata.shape
    # get randomly-positioned chunk of frames
    start_frame = random.randint(1,248)
    shorter = videodata[start_frame:start_frame+50,:,:,:]
    if orig_shape[1] != 1080:
        ret = np.swapaxes(shorter,1,2)
    else:
        ret = shorter
    return ret

def to_grayscale(point):
    grayscale = np.average(point, axis=3)
    return grayscale

def datapoints():
    filenames = glob.glob("train_sample_videos/*.mp4")
    metadata = read_metadata()
    for i, filename in enumerate(filenames):
        videodata = skvideo.io.vread(filename)
        point = slice_videodata(videodata)
        #point = to_grayscale(point)
        #point = np.expand_dims(point,-1)
        point = np.expand_dims(point,0)
        basename = os.path.basename(filename)
        label_str = metadata[basename]['label']
        if label_str == "FAKE":
            label = np.array([[1,0]])
        else:
            label = np.array([[0,1]])
        yield i, point, filename, label

