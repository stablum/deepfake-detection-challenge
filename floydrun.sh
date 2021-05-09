#!/bin/bash
. ./venv/bin/activate
floyd run --data frablum/datasets/train_sample_videos/1:train_sample_videos --gpu --env tensorflow-1.14 "ln -s /floyd/input/train_sample_videos ./train_sample_videos ; python3 run.py"
