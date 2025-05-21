import logging
import os
import sys
import csv
import cv2
import re

import torch
import glob

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


CLASS_LIST = {}
CLASS_LIST['bicycle'] = 1
CLASS_LIST['car'] = 2
CLASS_LIST['motorcycle'] = 3
CLASS_LIST['bus'] = 5
CLASS_LIST['truck'] = 7


def file_nums(file_name): #old
    # Extract digits using regular expression
    digits_match = re.search(r'\d+', file_name)

    #if digits_match:
    #    extracted_digits = digits_match.group()
    #    print(f"Extracted Digits: {extracted_digits}")
    #else:
    #    print("No digits found in the file name.")

    return digits_match.group() 
# def file_nums(file_name):

#     return file_name.split('_t')[1].split('_.png')[0]

def init_model():
    try:
        model = torch.hub.load('ultralytics/yolov5', 'custom', 'yolov5/models/yolov5s.pt')
    except:
        print("Can't find model")
        #model = torch.hub.load('ultralytics/yolov7', 'yolov7.pt')
    return model

def process_files(directory,save_dir,model):

    # List all files in the directory
    all_files = os.listdir(directory)

    # Filter files based on the criteria
    target_files = [file for file in all_files if file.startswith("img_t") and file.endswith("_.png")]

    # Sort the files to process them in order
    target_files.sort()

    
    for ii,file_name in enumerate(target_files):
        sys.stdout.write(f"Processing {ii+1}/{len(target_files)}")
        sys.stdout.flush()

        file_path = os.path.join(directory, file_name)
        dat = file_nums(file_name)

        try:
            results = model(file_path)
            
            # Replace 'your_image.jpg' with the path to your JPEG image
            frame = cv2.imread(file_path)#image_path)
            shape = frame.shape
            all = shape[0] * shape[1]

            # #calculate area of bboxes
            area = 0
            for pred in results.pandas().xyxy[0].values:
                xmin,ymin,xmax,ymax,confidence,cl,name = pred
                if name in CLASS_LIST.keys():
                    area += _getArea(xmin,ymin,xmax,ymax)

                    if ii%50==0: #50
                        x,y = int(xmin),int(ymin)
                        w,h = int(xmax-xmin), int(ymax-ymin)
                        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
                        file_name = f'{save_dir}/{dat}' +'.png'
                        cv2.imwrite(file_name, frame) 


            #density = 1 - float(area) / all
            density = float(area) / all
        

            #print("Density:",density)
        except:
            density = 0

        row = [str(dat), density]
        #with open(f'results/{subdir}.csv', "a") as file:
        with open(f"{save_dir}.csv","a") as file:
            writer = csv.writer(file)
            writer.writerow(row)

        #print("Result for {}: {}".format(file_name,density))

        # results.show        

def _getArea(box):
    return (box[2] - box[0]) * (box[3] - box[1])

def _getArea(xmin,ymin,xmax,ymax):
    return (xmax - xmin) * (ymax - ymin)
    
def main():
    # Check if a directory argument is provided
    if len(sys.argv) != 2:
        print("Usage: python script.py /path/to/your/directory")
        sys.exit(1)

    directory_path = sys.argv[1]
    #directory_path = "/path/to/your/directory"  # Replace with the actual path
    root_dir = directory_path.split('/')[-1] #day dir
    print("Processing:",root_dir)
    days = os.listdir(directory_path)
    model = init_model()
    for day in days[1:2]:
        cameras = glob.glob(directory_path + f'/{day}/Cam*')
        for camera in cameras:
            input_path = f"{camera}"
            sp=camera.split("../")[1]
            save_path = f"./results/{sp}"
            if not os.path.exists(save_path):
                log.debug(f"Creating image directory `{save_path}`...")
                os.makedirs(save_path)
            process_files(directory=input_path,save_dir = save_path,model=model)

    

if __name__ == "__main__":
    log = logging.getLogger("main")


    main()
