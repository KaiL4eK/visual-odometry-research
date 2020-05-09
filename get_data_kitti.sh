#!/usr/bin/env bash

mkdir -p data/KITTI; cd $_
wget -N https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_color.zip
wget -N https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_gray.zip
wget -N https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_poses.zip
wget -N https://s3.eu-central-1.amazonaws.com/avg-kitti/devkit_odometry.zip
wget -N https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_calib.zip
