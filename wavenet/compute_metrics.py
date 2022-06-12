# coding: utf-8
"""Compute MCD metric
Based on https://raw.githubusercontent.com/ttslr/python-MCD/master/MCD-DTW.py

usage: compute_metrics.py [options]

options:
    --generated-dir=<dir>       Path to generated wav files
    --generated-pattern=<name>  [default: LJ050-*_gen.wav]
    --reference-dir=<dir>       Path to reference wav files
    --reference-pattern=<name>  [default: LJ050-*_ref.wav]
    -h, --help                  Show help message.
"""
from docopt import docopt

from glob import glob
import os

import numpy as np
import librosa
from scipy.io import wavfile
from scipy.stats import sem
import pysptk
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw


def readmgc(filename):
    # all parameters can adjust by yourself :)
    sr, x = wavfile.read(filename)
    assert sr == 22050
    x = x.astype(np.float64)
    frame_length = 1024
    hop_length = 256  
    # Windowing
    frames = librosa.util.frame(x, frame_length=frame_length, hop_length=hop_length).astype(np.float64).T
    frames *= pysptk.blackman(frame_length)
    assert frames.shape[1] == frame_length 
    # Order of mel-cepstrum
    order = 25
    alpha = 0.41
    stage = 5
    gamma = -1.0 / stage

    # print(filename, 'frames', frames.shape, frames.min(), frames.max())
    mgc = pysptk.mgcep(frames, order, alpha, gamma, min_det=1e-300)
    mgc = mgc.reshape(-1, order + 1)
    print("mgc of {} is ok!".format(filename))
    return mgc


def compute_metrics(reference_dir, reference_pattern, generated_dir, generated_pattern):
    reference_pattern = os.path.join(reference_dir, reference_pattern)
    reference_paths = glob(reference_pattern)
    reference_paths.sort()

    synth_pattern = os.path.join(generated_dir, generated_pattern)
    synth_paths = glob(synth_pattern)
    synth_paths.sort()

    assert len(synth_paths) == len(reference_paths)

    _logdb_const = 10.0 / np.log(10.0) * np.sqrt(2.0)
    s = 0.0
    
    framesTot = 0

    all_s = []


    for filename1, filename2 in zip(reference_paths, synth_paths):
        print("Processing -----------{}, {}".format(filename1, filename2))
        
        try:
            mgc1 = readmgc(filename1)
        except RuntimeError as e:
            print("ERROR reading and converting reference wav file", filename1, ":", e)
            continue

        try:
            mgc2 = readmgc(filename2)
        except RuntimeError as e:
            print("ERROR reading and converting synthesized wav file", filename2, ":", e)
            continue
    
        x = mgc1
        y = mgc2


        distance, path = fastdtw(x, y, dist=euclidean)
    
        distance/= (len(x) + len(y))
        pathx = list(map(lambda l: l[0], path))
        pathy = list(map(lambda l: l[1], path))
        x, y = x[pathx], y[pathy]

        frames = x.shape[0]
        framesTot  += frames

        z = x - y
        err = np.sqrt((z * z).sum(-1))
        print(err.shape)
        s += err.sum()
        all_s.append(err)



    MCD_value = _logdb_const * float(s) / float(framesTot)
    all_s = _logdb_const * np.concatenate(all_s, axis=0)
    MCD_mean = all_s.mean()
    MCD_sem = sem(all_s)

    print("MCD = : \mcdmetric{"+str(MCD_value)+"}, \mcdmetric{"+str(MCD_mean)+"}{"+str(MCD_sem)+"}")

    return MCD_value, MCD_mean, MCD_sem


if __name__ == "__main__":
    args = docopt(__doc__)

    compute_metrics(args["--reference-dir"], args["--reference-pattern"],
                    args["--generated-dir"], args["--generated-pattern"])

