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
import keras
import time

def log_config():
    print(dir(config))
    for curr in dir(config):
        if curr[0] != '_':
            val = getattr(config, curr)
            mlflow.log_param(curr,val)
    mlflow.log_artifact('config.py')

def log_net(net):
    pretty_net = net.to_json(indent=4)
    structure_filename = tempfile.mkstemp(prefix="deepfake_net_structure_",suffix=".txt")[1]
    plot_filename= tempfile.mkstemp(prefix="deepfake_net_plot_",suffix=".png")[1]
    f = open(structure_filename,'w')
    f.write(pretty_net)
    f.flush()
    f.close()
    mlflow.log_artifact(structure_filename)
    keras.utils.plot_model(
        net,
        to_file=plot_filename,
        show_shapes=True,
        show_layer_names=True
        #expand_nested=True
    )
    mlflow.log_artifact(plot_filename)
    summary_filename = tempfile.mkstemp(prefix="deepfake_net_summary_",suffix=".txt")[1]
    f = open(summary_filename,'w')
    keras.utils.print_summary(net,print_fn=lambda line: f.write(line+'\n'))
    f.flush()
    f.close()
    mlflow.log_artifact(summary_filename)

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
    print("start_frame", start_frame)
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
    #filenames = glob.glob("/mnt/ramdisk/*.mp4")
    filenames = glob.glob("train_sample_videos/*.mp4")
    random.shuffle(filenames) # always load datapoint in different order
    metadata = read_metadata()
    for i, filename in enumerate(tqdm.tqdm(filenames[:config.points_per_epoch])):
        start = time.time()
        videodata = skvideo.io.vread(filename)
        end_load = time.time()
        print("loading time:",end_load-start)
        basename = os.path.basename(filename)
        label_str = metadata[basename]['label']
        if label_str == "FAKE":
            label = np.array([[1,0]])
            n_slices = 5
        else:
            label = np.array([[0,1]])
            n_slices = 21

        slices = []

        for _ in range(n_slices):
            curr = slice_videodata(videodata)
            #point = to_grayscale(point)
            #point = np.expand_dims(point,-1)
            curr = np.expand_dims(curr,0)
            slices.append(curr)
        end = time.time()
        print("elapsed time to load and process",filename,":",end - start)
        yield i,slices, filename, label

