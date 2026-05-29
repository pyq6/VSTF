from ultralytics import YOLO
import clip
import os
import sys
import math
import torch
import numpy as np

if __name__ == "__main__":
    model = YOLO("ultralytics/cfg/models/11/yolo11s.yaml")
 
    model.train(
       # resume=True, 
        data="Dataset/dataset_sort.yaml",
        pretrained=False,  
        imgsz=640,
        epochs=150,                      
        batch=32,
        close_mosaic=10,
        workers=6,
        device="0", 
        
     
        optimizer="SGD",         
    
        lr0=0.005,
        momentum=0.937,          
        weight_decay=0.0005,    
        warmup_epochs=15,      
        cos_lr=True,            
        
        amp= True ,
        cache=True,
        project="runs/xx",
        name="11s",
        resume=False,
        fraction=1.0,
        seed=42,     
        copy_paste=0.2,
    )