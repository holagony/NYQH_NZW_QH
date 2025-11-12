# !/usr/bin/env python
# coding:UTF-8

# 将py文件编译成pyd文件，需要安装VS


# Use function_to_be_profiled as intended

import os
import shutil
import glob
import sys


def compileSingleFile(file, pyfile_delete=False):
    """

    :param file: str, py文件
    :param pyfile_delete: bool, 是否删除py文件
    :return:
    """
    try:
        outdir = os.path.dirname(file)
        filename = os.path.basename(file).split(".")[0]
        os.chdir(outdir)
        os.system("easycython "+file)
        # f = os.popen("easycython "+file)
        # f.readlines()
        shutil.rmtree(os.path.join(outdir, "build"))
        os.remove(os.path.join(outdir, filename+".c"))
        os.remove(os.path.join(outdir, filename + ".html"))
        pydfile = glob.glob(os.path.join(outdir, filename+"*.pyd"))[0]
        outfile = os.path.join(outdir, filename+".pyd")
        os.rename(pydfile, outfile)
        if pyfile_delete:
            os.remove(file)
    finally:
        pass

#file =r"C:\Users\LYB\Desktop\pyd_灾害\SZRainStormDanger\common_tool\com_tool.py"
#compileSingleFile(file, pyfile_delete=True)
#
if __name__ == "__main__":
    
    indir = r"C:\Users\liyaobin\Desktop\gw_preprocess"
    # indir = sys.argv[1]
    # # files = glob.glob(os.path.join(indir, "*.py"))
    filelist = []
    for dirpath, dirnames, filenames in os.walk(indir):
        for filename in filenames:
            if filename.split(".")[1]=="py":
                filelist.append(os.path.join(dirpath, filename))
    for file in filelist:
        compileSingleFile(file, pyfile_delete=True)


