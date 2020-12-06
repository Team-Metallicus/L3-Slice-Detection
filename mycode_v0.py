import os
import sys, getopt

import uuid

import SimpleITK as sitk
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, flash, request, redirect, render_template
from flask import jsonify
from flask import send_from_directory
from flask_materialize import Material
from tensorflow.python.keras.backend import set_session
from werkzeug.utils import secure_filename



#from sarcopenia_ai.apps.segmentation.segloader import preprocess_test_image
from sarcopenia_ai.apps.server import settings
from sarcopenia_ai.apps.slice_detection.predict import parse_inputs, to256
from sarcopenia_ai.apps.slice_detection.utils import decode_slice_detection_prediction, \
    preprocess_sitk_image_for_slice_detection, adjust_detected_position_spacing, place_line_on_img
from sarcopenia_ai.core.model_wrapper import BaseModelWrapper
from sarcopenia_ai.io import load_image
from sarcopenia_ai.preprocessing.preprocessing import blend2d
from sarcopenia_ai.utils import compute_muscle_area, compute_muscle_attenuation


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
graph = tf.get_default_graph()





#Read arguments
#############################
import argparse 
  
msg = "Adding description"
      
# Initialize parser 
parser = argparse.ArgumentParser(description = msg) 

# Adding optional argument 
parser.add_argument("-i", "--Input", help = "Input file or folder") 
#parser.add_argument("-o", "--Output", help = "Show Output") 

# Read arguments from command line 
args = parser.parse_args() 
  
#if args.Output: 
#    print("Diplaying Output as: % s" % args.Output) 
#    parser.parse_args()



#outputfile = args.Output
#print('Output file is "', outputfile)


#load_model
##############################
# set_session(sess)
model_wrapper = BaseModelWrapper(settings.SLICE_DETECTION_MODEL_PATH)
model_wrapper.setup_model()
global slice_detection_model
slice_detection_model = model_wrapper.model
slice_detection_model._make_predict_function()

global segmentation_model
model_wrapper = BaseModelWrapper(settings.SEGMENTATION_MODEL_PATH)
model_wrapper.setup_model()
segmentation_model = model_wrapper.model
segmentation_model._make_predict_function()
    # global graph
    # graph = tf.get_default_graph()


    #process_file
    ###############################

    #global segmentation_model
    #global slice_detection_model

    #image_path = 'data/volume.nii'
import os
directory = args.Input
import pandas as pd

pred_id = 0
cols = ['Folder_Path','Patient_Folder','Study_Folder','Serie_Folder','L3_detection','L3_position','Confidence','Output_filename1','Output_filename2','Error'  ]
lst = []

for folder in os.listdir(directory):
    print("IN FOLDER : "+folder)
    if(folder=='.DS_Store'):
        continue
    for sub_folder in os.listdir(directory+"/"+folder):
        if(sub_folder=='.DS_Store'):
            continue
        print("IN SUB-FOLDER: "+sub_folder)
        for sub_sub_folder in os.listdir(directory+"/"+folder+"/"+sub_folder):
            for file in os.listdir(directory+"/"+folder+"/"+sub_folder+"/"+sub_sub_folder):
                #print(file)
                if(file.endswith(".nii") or file.endswith(".nii.gz")):
                    print("Processing file: "+file)
                    try:
        
                        if(sub_sub_folder=='.DS_Store'):
                            continue
                        print("IN SUB-SUB-FOLDER: "+sub_sub_folder)
            
                        image_path = directory+"/"+folder+"/"+sub_folder+"/"+sub_sub_folder+"/"+file
                        prob_threshold=0.1
            
            
                        print(image_path)
            
                        #gething the image name only withou path
                        import ntpath
                        head, tail =  ntpath.split(image_path)
                        image_name =  tail or ntpath.basename(head)
            
                        pred_id = pred_id +1
                        results = {"success": False, "prediction": {'id': pred_id}}
            
                        
                        sitk_image, _ = load_image(image_path)
    
                        image2d, image2d_preview = preprocess_sitk_image_for_slice_detection(sitk_image)
                        image3d = sitk.GetArrayFromImage(sitk_image)
            
                        spacing = sitk_image.GetSpacing()
                        size = list(sitk_image.GetSize())
            
                        with graph.as_default():
                            set_session(sess)
                            preds = slice_detection_model.predict(image2d)
            
                        pred_z, prob = decode_slice_detection_prediction(preds)
                        slice_z = adjust_detected_position_spacing(pred_z, spacing)
            
                        slice_detected = prob > prob_threshold
                        #print(slice_z)
                        if slice_detected:
                            results["prediction"]["slice_z"] = slice_z
                            results["prediction"]["slice_prob"] = '{0:.2f}%'.format(100 * prob)
                            results["success"] = True
            
                            #creating the (2) jpeg imageS of slice position and L3 slice
                            #######################################
                            image = image3d
                            slice_image = image[slice_z, :, :]
                            image2d = place_line_on_img(image2d[0], pred_z, pred_z, r=1)

                            cv2.imwrite(str(pred_id)+'_YES_'+image_name+'_SL_'+str(slice_z)+'_PROB_'+str(prob)+'.jpg', to256(slice_image))
                            cv2.imwrite(str(pred_id)+'_YES_'+image_name+'_FRL_'+str(slice_z)+'_PROB_'+str(prob)+'.jpg', to256(image2d))
                            
                            output = [image_path,folder,sub_folder,sub_sub_folder,'YES',slice_z,prob,str(pred_id)+'_YES_'+image_name+'_SL_'+str(slice_z)+'_PROB_'+str(prob)+'.jpg',str(pred_id)+'_YES_'+image_name+'_FR_'+str(slice_z)+'_PROB_'+str(prob)+'.jpg','']
                            lst.append(output)
            
                        if results["success"]:
                            result_str = 'Slice detected at position {0} of {1} with {2:.2f}% confidence '.format(slice_z, size[2],
                                                                                                                          100 * prob)
                            print(result_str)
                        else:
                            result_str = 'Slice not detected'
                            image = image3d
                            slice_image = image[slice_z, :, :]
                            image2d = place_line_on_img(image2d[0], pred_z, pred_z, r=1)

                            cv2.imwrite(str(pred_id)+'_NO_'+image_name+'_SL_'+str(slice_z)+'_PROB_'+str(prob)+'.jpg', to256(slice_image))
                            cv2.imwrite(str(pred_id)+'_NO_'+image_name+'_FR_'+str(slice_z)+'_PROB_'+str(prob)+'.jpg', to256(image2d))
                            
                            output = [image_path,folder,sub_folder,sub_sub_folder,'NO',slice_z,prob,str(pred_id)+'_NO_'+image_name+'_SL_'+str(slice_z)+'_PROB_'+str(prob)+'.jpg',str(pred_id)+'_NO_'+image_name+'_FR_'+str(slice_z)+'_PROB_'+str(prob)+'.jpg','']
                            lst.append(output)
            
                        results["prediction"]["str"] = result_str
                        #print(result_str)
        
                    #except ValueError:
                        #print("ValueError - File: "+directory+"/"+folder+"/"+sub_folder)
                    #    output = [image_path,folder,sub_folder,sub_sub_folder,'Error',slice_z,prob,'','','ValueError']
                    #    lst.append(output)
                    except:
                        #print("Something else went wrong - File: "+directory+"/"+folder+"/"+sub_folder)
                        output = [image_path,folder,sub_folder,sub_sub_folder,'Error','','','','','Something went wrong']
                        lst.append(output)
                        raise

import pandas as pd
df = pd.DataFrame(lst, columns=cols)
df.to_csv("algo1_output_v0.csv")
