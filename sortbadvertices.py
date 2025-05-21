import logging
import os
import sys
import csv
import cv2
import re
import numpy as np
import matplotlib.pyplot as plt
import datetime
import glob
import pandas as pd


import mplhep as hep

# hep.style.use(hep.style.ALICE)
hep.style.use(hep.style.ROOT)


def main(fFullData=True):
    path = "/Users/joachimcarlokristianhansen/XIMANTIS/locald/cosmos_data/May2020/0517"
    #!ls {path}

    tmp = glob.glob(path + '/' + "Cam*_aggregated.csv")
    cam_ids = [i.split('/')[-1].split('_aggregated.csv')[0] for i in tmp]

    cam_oi = []


    for ii,cam in enumerate(cam_ids):
        #if ii in bad_vertices:
        #   continue
        cam_folder = path + f"/{cam}"
        nr = int(cam.split("Cam")[-1])
        if os.listdir(cam_folder):
            cam_oi.append(nr)
    l1,l2 = len(cam_oi),len(cam_ids)
    cam_oi.sort()



    cam_oi2 = []
    if fFullData:
        bad_vertices = []
    else:
        bad_vertices=[4,8,9,10,12,30,79,100,101,117]
    for cam in cam_oi:
        if cam in bad_vertices:
            continue
        else:
            cam_oi2.append(cam)
    l3 = len(cam_oi2)
    print(f"Using a total of {l3}/{l2} cameras")

    pd.DataFrame({"camera":cam_oi2}).to_csv("gps_data/coi.csv",index=False)


if __name__=="__main__":

    main()