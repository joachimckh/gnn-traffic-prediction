import os,sys
import numpy as np
import matplotlib.pyplot as plt
import datetime
import glob
import pandas as pd

from BuildNodes import fovpoint

# def euclidean_dist(df1,df2):
#     y_dist = df2.y - df1.y
#     x_dist = df2.x - df1.x
#     return np.sqrt( x_dist**2 + y_dist**2 )


def main():

    #id,x,y,bearing

    #df = pd.read_csv("gps_data/GPScleaned.csv",header=0)
    df = pd.read_csv("gps_data/CamerasForUse_Modified.csv",header=0)
    coi = pd.read_csv("gps_data/coi.csv",header=0)
    camera_of_interest = coi.camera.values
    print("coi",camera_of_interest)
    print("coi len",len(camera_of_interest))
    len_df = len(df.index)

    print("Filtering out bad vertices")
    print("Length of GPS data:",len_df)
    df = df[df["id"].isin(camera_of_interest)]
    len_df = len(df)
    print("New length:",len(df))

    # filter out based on what I chose in NodeConnection (fUseCam)
    mask = df['fUseCam']==1
    df_use = df[mask]
    len_df = len(df_use)
    print("Length with fUseCam 1:", len_df)



    _from,_to,_cost = [],[],[]

    for i in range(len_df):
        tmp_i = df_use.iloc[i]
        for j in range(len_df):
            if i==j:
                continue
            
            tmp_j = df_use.iloc[j]
            point = fovpoint(tmp_i)
            fFov = point.within_field_of_view(tmp_j)
            if fFov:
                #tmp = euclidean_dist(tmp_i,tmp_j)
                _from.append(int(i))
                _to.append(int(j))
                _cost.append(point.dist)

    print("length of selected")
    print(len(_from))
    out = pd.DataFrame({'from':_from,'to':_to,'cost':_cost})

    out.to_csv("gps_data/tofromcost.csv",index=False)
    df_use.to_csv("gps_data/finalcamerasforuse.csv",index=False)
                
            

    return 0x0



if __name__=="__main__":

    main()