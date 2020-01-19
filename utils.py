import skvideo.io
import humanize
import glob
import numpy as np
import json
import os
import random
import tqdm
import config
import mlflow
import tempfile

def log_config():
    print(dir(config))
    for curr in dir(config):
        if curr[0] != '_':
            val = getattr(config, curr)
            mlflow.log_param(curr,val)

def log_net(net):
    pretty_net = net.to_json(indent=4)
    filename = tempfile.mkstemp(prefix="deepfake_net_",suffix=".txt")[1]
    f = open(filename,'w')
    f.write(pretty_net)
    f.flush()
    f.close()
    mlflow.log_artifact(filename)

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
    shorter = videodata[start_frame:start_frame+config.frames_per_point,:,:,:]
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
    random.shuffle(filenames) # always load datapoint in different order
    metadata = read_metadata()
    for i, filename in enumerate(tqdm.tqdm(filenames[:config.points_per_epoch])):
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

