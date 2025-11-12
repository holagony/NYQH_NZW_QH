#!/usr/bin/env bash

# source ~/.bashrc
conda activate py3

workspace_dir="/mnt/PRESKY/user/wuyujie/yanfa/NYQH/NYQH_NZW_QH/"
cd $workspace_dir
main_fuc='app.py'
python ${workspace_dir}${main_fuc} $1