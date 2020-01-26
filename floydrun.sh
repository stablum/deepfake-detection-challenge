#!/bin/bash
floyd run --data frablum/datasets/train_sample_videos:train_sample_videos --gpu --env tensorflow-1.14 python3 run.py
