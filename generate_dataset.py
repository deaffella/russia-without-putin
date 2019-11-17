import argparse
import subprocess
import ntpath
import os

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
  help="path to input video")
ap.add_argument("-o", "--output", required=True,
  help="path to output dataset folder")
args = vars(ap.parse_args())

basename = ntpath.basename(args["input"])
basename = os.path.splitext(basename)[0]

subprocess.call(['ffmpeg', '-i', args["input"], args["output"] + '/' + basename + '-%05d.jpg'])