import os
import csv
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



##from sarcopenia_ai.apps.segmentation.segloader import preprocess_test_image
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



##from sarcopenia_ai.apps.segmentation.segloader import preprocess_test_image
## TO FIX ---
##Traceback (most recent call last):
##  File "mycode.py", line 19, in <module>
##    from sarcopenia_ai.apps.segmentation.segloader import preprocess_test_image
##  File "/content/gdrive/My Drive/sarcopenia-ai-master/sarcopenia_ai/apps/segmentation/segloader.py", line 8, in <module>
##    from midatasets.preprocessing import normalise_zero_one, normalise_one_one
##ModuleNotFoundError: No module named 'midatasets'
##
import cv2
import numpy as np
def normalise_zero_one(image, eps=1e-8):
    print("Here 1")
    image = image.astype(np.float32)
    ret = (image - np.min(image))
    ret /= (np.max(image) - np.min(image) + eps)
    return ret

def normalise_one_one(image):
    print("Here 2")
    ret = normalise_zero_one(image)
    ret *= 2.
    ret -= 1.
    return ret
    
def preprocess_test_image(image):
    print("Here")
    #image = normalise_one_one(image, -250, 250)
    image = normalise_one_one(image)
    return image
##################    


def find_max(img):
    return np.unravel_index(np.argmax(img, axis=None), img.shape)[0]

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

#newly add    
def reduce_hu_intensity_range(img, minv=100, maxv=1500):
    img = np.clip(img, minv, maxv)
    img = 255 * normalise_zero_one(img)

    return img


import os
directory = args.Input
import pandas as pd

pred_id = 0
cols = ['Folder_Path','Patient_Folder','Study_Folder','Serie_Folder','L3_detection','L3_position','Total_slices','Confidence','Output_filename1','Output_filename2','Error'  ]
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
                print("IN SUB-SUB-FOLDER: "+sub_sub_folder)
                #print(file)
                if(file.endswith(".nii.gz") or file.endswith(".nii")):
                    print("Processing file: "+file)
                    try:
        
                        if(sub_sub_folder=='.DS_Store'):
                            continue
                        print("IN SUB-SUB-FOLDER: "+sub_sub_folder)
            
                        image_path = directory+"/"+folder+"/"+sub_folder+"/"+sub_sub_folder+"/"+file
                        
                        prob_threshold_U=0.12
                        prob_threshold_L=0.03
            
            
                        print(image_path)
            
                        #gething the image name only withou path
                        import ntpath
                        head, tail =  ntpath.split(image_path)
                        image_name =  tail or ntpath.basename(head)
            
                        pred_id = pred_id +1
                        results = {"success": False, "prediction": {'id': pred_id}}
            
                        
                        sitk_image, _ = load_image(image_path)

                        #sitk_image = np.flipud(sitk_image)
                        
                        image2d, image2d_preview = preprocess_sitk_image_for_slice_detection(sitk_image)
                        image3d = sitk.GetArrayFromImage(sitk_image)
                        
                        print(image3d.shape)
                        print(image2d.shape)
            
                        spacing = sitk_image.GetSpacing()
                        size = list(sitk_image.GetSize())
                        
                        
            
                        with graph.as_default():
                            set_session(sess)
                            #TO-DO: Process image2d to normalize HU and adapt for full body ct
                            preds = slice_detection_model.predict(image2d)
            
                        pred_z, prob = decode_slice_detection_prediction(preds)
                        slice_z = adjust_detected_position_spacing(pred_z, spacing)
                        
                        #Find local max from 27% to 48% of the body image
                        new_z_calculate = 0;
                        new_max_z = pred_z;
                        new_slice_z = slice_z;
                        
                        if(slice_z < .27*size[2] or slice_z < .48*size[2]):
                            print("debug")
                            print(preds.shape)
                            print(preds.shape[1])
                            new_pred_z = find_max(preds[0, int(.27*preds.shape[1]):int(.48*preds.shape[1])])
                            new_pred_z = new_pred_z + int(.27*preds.shape[1]);
                            new_slice_z = adjust_detected_position_spacing(new_pred_z, spacing)
                            print("old position")
                            print(pred_z)
                            print(slice_z)
                            print("new position")
                            print(new_pred_z)
                            print(new_slice_z)
                            new_z_calculate =1;
                            #prob = float(preds[new_max_z]])
 
                        print("Prediction_Values")
                        #print(preds)
                        print(type(preds))
                        import pandas as pd
                        ## convert your array into a dataframe
                        import numpy
                        #a = numpy.asarray([ [1,2,3], [4,5,6], [7,8,9] ])
                        # reshaping the array from 3D 
                        # matrice to 2D matrice. 
                        preds_reshaped = preds.reshape(preds.shape[0], -1) 
                        numpy.savetxt(image_path+".csv", preds_reshaped, delimiter=",")
                        print("Pred_Z")
                        print(pred_z)
                        #print("Slize_Z")
                        #print(slice_z);
            
                        #slice_detected = prob > prob_threshold_U
                        #slice_not_detected = prob < prob_threshold_L
                        
                        #print(slice_z)
                        if (prob > prob_threshold_U):
                            #creating the (2) jpeg imageS of slice position and L3 slice
                            #######################################
                            image = image3d
                            slice_image = image[new_slice_z, :, :]
                            image2dA = place_line_on_img(image2d[0], pred_z, pred_z, r=1)
                            image2dB = place_line_on_img(image2d[0], new_pred_z, new_pred_z, r=1)

                            
                            
                            
                            #with graph.as_default():
                            #    set_session(sess)
                            #    seg_image = segmentation_model.predict(preprocess_test_image(slice_image[np.newaxis, :, :, np.newaxis]))
                            #    seg_image = seg_image[0]
                                
                            #out_seg_image = np.flipud(blend2d(np.squeeze(slice_image),
                            #              np.squeeze(seg_image > 0.5), 0.5))
                             
                            cv2.imwrite(str(pred_id)+'_YES_'+image_name+'_SL_'+str(new_slice_z)+'_PROB_'+str(prob)+'.jpg', to256(slice_image))
                            #cv2.imwrite(str(pred_id)+'_YES_'+image_name+'_FR_'+str(slice_z)+'_PROB_'+str(prob)+'.jpg', to256(image2dA))
                            cv2.imwrite(str(pred_id)+'_YES_'+image_name+'_FR_'+str(new_slice_z)+'_PROB_'+str(prob)+'.jpg', to256(image2dB))
                            #cv2.imwrite(str(pred_id)+'_YES_'+image_name+'_SEG_'+str(slice_z)+'_PROB_'+str(prob)+'.jpg', to256(np.squeeze(out_seg_image)))
                            
                            output = [image_path,folder,sub_folder,sub_sub_folder,'YES',slice_z,size[2],prob,str(pred_id)+'_YES_'+image_name+'_SL_'+str(slice_z)+'_PROB_'+str(prob)+'.jpg',str(pred_id)+'_YES_'+image_name+'_FR_'+str(slice_z)+'_PROB_'+str(prob)+'.jpg','']
                            lst.append(output)
            
                            result_str = 'Slice detected at position {0} of {1} with {2:.2f}% confidence '.format(slice_z, size[2],
                                                                                                                          100 * prob)
                            print(result_str)
                        
                        elif (prob < prob_threshold_L):
                            #creating the (2) jpeg imageS of slice position and L3 slice
                            #######################################
                            image = image3d
                            slice_image = image[slice_z, :, :]
                            image2dA = place_line_on_img(image2d[0], pred_z, pred_z, r=1)
                            image2dB = place_line_on_img(image2d[0], new_pred_z, new_pred_z, r=1)

                            
                            #with graph.as_default():
                            #    set_session(sess)
                            #    seg_image = segmentation_model.predict(preprocess_test_image(slice_image[np.newaxis, :, :, np.newaxis]))
                            #    seg_image = seg_image[0]
                                
                            #out_seg_image = np.flipud(blend2d(np.squeeze(preprocess_test_image(slice_image)),
                            #              np.squeeze(seg_image > 0.5), 0.5))

                            cv2.imwrite(str(pred_id)+'_NO_'+image_name+'_SL_'+str(new_slice_z)+'_PROB_'+str(prob)+'.jpg', to256(slice_image))
                            #cv2.imwrite(str(pred_id)+'_NO_'+image_name+'_FR_'+str(slice_z)+'_PROB_'+str(prob)+'.jpg', to256(image2dA))
                            cv2.imwrite(str(pred_id)+'_NO_'+image_name+'_FR_'+str(new_slice_z)+'_PROB_'+str(prob)+'.jpg', to256(image2dB))
                            output = [image_path,folder,sub_folder,sub_sub_folder,'NO',slice_z,size[2],prob,str(pred_id)+'_NO_'+image_name+'_SL_'+str(slice_z)+'_PROB_'+str(prob)+'.jpg',str(pred_id)+'_NO_'+image_name+'_FR_'+str(slice_z)+'_PROB_'+str(prob)+'.jpg','']
                            lst.append(output)
            
                            result_str = 'L3 Slice not detected. Most likely to be L3 would be at position {0} of {1} with {2:.2f}% confidence '.format(slice_z, size[2],
                                                                                                                          100 * prob)
                            print(result_str)
                            
                            
                        else:
                            image = image3d
                            slice_image = image[slice_z, :, :]
                            image2dA = place_line_on_img(image2d[0], pred_z, pred_z, r=1)
                            image2dB = place_line_on_img(image2d[0], new_pred_z, new_pred_z, r=1)

                            
                            #with graph.as_default():
                            #    set_session(sess)
                            #    seg_image = segmentation_model.predict(preprocess_test_image(slice_image[np.newaxis, :, :, np.newaxis]))
                            #    seg_image = seg_image[0]
                                
                            #out_seg_image = np.flipud(blend2d(np.squeeze(preprocess_test_image(slice_image)),
                            #              np.squeeze(seg_image > 0.5), 0.5))

                            cv2.imwrite(str(pred_id)+'_MAYBE_'+image_name+'_SL_'+str(new_slice_z)+'_PROB_'+str(prob)+'.jpg', to256(slice_image))
                            #cv2.imwrite(str(pred_id)+'_MAYBE_'+image_name+'_FR_'+str(slice_z)+'_PROB_'+str(prob)+'.jpg', to256(image2dA))
                            cv2.imwrite(str(pred_id)+'_MAYBE_'+image_name+'_FR_'+str(new_slice_z)+'_PROB_'+str(prob)+'.jpg', to256(image2dB))
                            
                            #cv2.imwrite(str(pred_id)+'_MAYBE_'+image_name+'_SEG_'+str(slice_z)+'_PROB_'+str(prob)+'.jpg', to256(np.squeeze(out_seg_image)))
                            
                            output = [image_path,folder,sub_folder,sub_sub_folder,'MAYBE',slice_z,size[2],prob,str(pred_id)+'_MAYBE_'+image_name+'_SL_'+str(slice_z)+'_PROB_'+str(prob)+'.jpg',str(pred_id)+'_MAYBE_'+image_name+'_FR_'+str(slice_z)+'_PROB_'+str(prob)+'.jpg','']
                            lst.append(output)
            

                    #except ValueError:
                        #print("ValueError - File: "+directory+"/"+folder+"/"+sub_folder)
                    #    output = [image_path,folder,sub_folder,sub_sub_folder,'Error',slice_z,prob,'','','ValueError']
                    #    lst.append(output)
                    except:
                        print("Something went wrong - File: "+directory+"/"+folder+"/"+sub_folder)
                        print("Unexpected error"+str(sys.exc_info()[0]))
                        output = [image_path,folder,sub_folder,sub_sub_folder,'Error','','','','','','Something went wrong']
                        lst.append(output)
                        raise

import pandas as pd
df = pd.DataFrame(lst, columns=cols)
df.to_csv("algo_agd1_output.csv")
