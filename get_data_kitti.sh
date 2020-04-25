#!/usr/bin/env bash

mkdir -p data/KITTI; cd $_
wget -N http://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8/chess.zip
wget -N https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_poses.zip
wget -N https://s3.eu-central-1.amazonaws.com/avg-kitti/devkit_odometry.zip
