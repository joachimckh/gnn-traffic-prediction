import os,sys
import numpy as np
import matplotlib.pyplot as plt
import datetime
import glob
import pandas as pd
import datetime

names = {0: "time", 1: "flow"}

def read_dldens_csv(path):
    df = pd.read_csv(path,header=None,converters={0: str})
    df.rename(columns=names,inplace=True)
    return df

def read_aggregated_csv(path):
    df = pd.read_csv(path,header=0)
    #arr = df.values.squeeze()
    return df

def find_time(df,time,vDelta = 1):
    delta = datetime.timedelta(hours = vDelta)
    
    t = pd.DataFrame()
    counter = 0
    time = (datetime.datetime.combine(datetime.date(1,1,1),time) + delta).time()
    while t.empty:
        t = df[df['time'].isin([time.strftime("%H") + "{:02d}".format(counter)])] 
        if t.empty:
            #print("time could not be found at:",time.strftime("%H"))
            #print("Recursively trying by increasing minute counter by +1")
            counter+=1
            
        if counter>30:
            raise ValueError("Horse")
            
    
    return t.index[0],time

def find_groups(df,n_splits=10):
    overall_mean=[]
    time_step =[]
    
    time = datetime.time(hour=int("05")) # start tiem
    
    # 0: 05-06 .. 1: 06-07 .. 2: 07-08 .. 3: 08-09 .. 4: 09-10 .. 5: 10-11 .. 6: 11-12 .. 7: 12-13
    # 8: 13-14 .. 8: 14-15 .. 9: 15-16 . 10: 16-17 . 11: 17-18 . 12: 18-19 . 13: 19-20 . 14: 20-21
    for _time in range(15): #16 timestep (from 5-21)
        start_index,time = find_time(df,time,vDelta=0)
        tmp = time
        end_index,time = find_time(df,time,vDelta=1)
        #print(f"from time {tmp.strftime('%H')}-{time.strftime('%H')}")
        #print(f"indices: {start_index} : {end_index}")

        tmp = df.values[start_index:end_index][:,1]


        data = np.array_split(tmp,n_splits)
        mean=[]
        for row in data:
            mean.append(row.mean())
        mean = np.array(mean)

        overall_mean.append(mean)
        time_step.append(np.ones_like(mean)*_time)

    overall_mean = np.array(overall_mean)
    time_step = np.array(time_step)
    return overall_mean.flatten(), time_step.flatten()



def Aggregate(path):
    #path = "results/0514/Cam117.csv"
    out_file = path.split('.csv')[0] + "_aggregated.csv"
    df = read_dldens_csv(path)
    m,time = find_groups(df)
    pd.DataFrame({'density':m,'time':time.astype(int)}).to_csv(out_file,index=False)


def Combine():
    path = "results/0514/Cam117_test_aggregated.csv"
    read_aggregated_csv(path)


def main(fType=1234,fMON="June"):
    path = f"../cosmos_data/{fMON}2020"
    if fMON=="May":
        day_skip = ["0513","0520"]
    elif fMON=="June":
        day_skip = ["0615"]
    elif fMON=="July":
        day_skip = ["0704","0706","0707"]
    else:
        day_skip = []

    camera_ids = pd.read_csv("gps_data/finalcamerasforuse.csv",header=0).id.values

    days = sorted(os.listdir(path))
    year = 2020
    
    if fType==0:
        for day in days:
            print("\rDay:",day)
            if day in day_skip:
                continue
            tmp_path = path + "/" + day
            for id in camera_ids:
                sys.stdout.write("\rAGGREGATE Processing id {} for day: {}".format(id,day))
                sys.stdout.flush()

                csv_file = tmp_path + f"/Cam{id}.csv"
                Aggregate(csv_file)

            
        
    
    else:
        agg_arr = [[] for _ in range(len(camera_ids))]

        count = 0
        for ik,day in enumerate(days):
            weekday = datetime.datetime(year,int(day[:2]),int(day[2:])).weekday()
            if day in day_skip:
                continue
            tmp_path = path + "/" + day
            for ii,id in enumerate(camera_ids):


                
                sys.stdout.write("\rCOMBINE Processing id {} for day: {}".format(id,day))
                sys.stdout.flush()

                csv_file = tmp_path + f"/Cam{id}_aggregated.csv"
                #tmp_arr = read_aggregated_csv(csv_file)
                df = read_aggregated_csv(csv_file)
                tmp_arr = df.density.values
                time_arr = df.time.values
                tmp_shape=tmp_arr.shape

                
                wd = np.ones_like(tmp_arr.flatten())*weekday
                
                agg_arr[ii].append([tmp_arr.flatten(),time_arr.flatten(),wd])
                
                
            count+=1
                    

        vertices = len(camera_ids) #64
        features = 2
        agg_arr = np.array(agg_arr)
        print("\n")
        print("#"*25)
        print(agg_arr.shape)
        print("Vertices:",vertices)
        print("Features:",features)

        print("#"*25)

        sshape = agg_arr.shape
        # agg_arr = agg_arr.transpose(3,1,0,2).reshape(sshape[1]*sshape[3],vertices,features)
        print("\nArray shape:",agg_arr.shape)
        #ytmp = np.stack(agg_arr,axis=1)
        #yout = np.expand_dims(ytmp,axis=2)
        #print("\Out shape:",yout.shape)
        np.savez(f'processed_data/{fMON}2020.npz', data=agg_arr,)
            


def ConcatenateAll():
    may = np.load("processed_data/May2020.npz")["data"]
    june = np.load("processed_data/June2020.npz")["data"]
    july = np.load("processed_data/July2020.npz")["data"]
    print(may.shape)
    print(june.shape)
    print(july.shape)
    all = np.concatenate([may,june,july],axis=1)
    print("#"*25)
    print("Data Shape\n")
    print(all.shape)
    print("#"*25)

    # all = all[...,:].reshape(24,64,15,30).transpose(0,3,1,2).reshape(24*30,64,15) #  

    #normally this one
   # all = all.transpose(1,3,0,2).reshape(27*150,all.shape[0],1) # for S (+) V (+) F

    #for donglin
    all = all.transpose(1,0,2,3)
    print("reshaped data")
    print(all.shape)

    np.savez(f'processed_data/All2020.npz', data=all,)


if __name__=="__main__":
    
    selection = sys.argv[1]

    if len(sys.argv)>2:
        num = int(sys.argv[2])
    else:
        num=123

    if selection=="concat":
        ConcatenateAll()    
    elif selection in ["May","June","July"]:
        main(fType=num,fMON=selection)
    elif selection in ["all"]:
        for ddayy in ["May","June","July"]:
            main(fType=num,fMON=ddayy) 
    else:
        print("select correct input")
        sys.exit(1)
    