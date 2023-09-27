from django.shortcuts import render
from django.http import JsonResponse, StreamingHttpResponse
from django.http import HttpResponse
# Create your views here.
from .torch_inference import infer
from django.views.decorators.csrf import csrf_exempt
from django.core.serializers.json import DjangoJSONEncoder
import numpy as np
import cv2.dnn
import cv2
from ultralytics.utils import ROOT, yaml_load
from ultralytics.utils.checks import check_yaml
import os
import tempfile
import subprocess
from roboflow import Roboflow
from ultralytics import YOLO
import shutil
import random
import yaml
from .anomaly_training import train
from argparse import Namespace


  ############################################################################################################################
                                      
                                       ######### Yolo Inference ##############


def yolo(image_bytes, model_path):
    try:
        # Load the YOLO model from the specified ONNX file
        model = YOLO(model_path)

        image_array = np.frombuffer(image_bytes, np.uint8)
        original_image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        result = model(original_image)
        res_plotted = result[0].plot()

        return res_plotted

    except Exception as e:
        return {'error': str(e)}


def yolo_api(request):
    if request.method == 'POST':
        if 'image' in request.FILES and 'model_path' in request.POST:
            image = request.FILES['image'].read()
            model_path = request.POST['model_path']
             
            print('Received Model Path:', model_path)  # Debugging print

            try:
                detections = yolo(image, model_path)

                # Save the annotated image to a temporary file
                temp_output_path = os.path.join(tempfile.gettempdir(), 'output.jpg')
                cv2.imwrite(temp_output_path, detections)

                # Return the annotated image as a response
                with open(temp_output_path, 'rb') as image_file:
                    response = HttpResponse(image_file.read(), content_type='image/jpeg')
                    response['Content-Disposition'] = 'inline; filename=output.jpg'
                    return response
            except Exception as e:
                return JsonResponse({'error': str(e)}, status=500)
        else:
            return JsonResponse({'error': 'Image or model_path not provided'}, status=400)
    else:
        return JsonResponse({'error': 'Only POST requests are allowed'}, status=405)


#############################################################################################################################
                                        
                                        ##### Yolo Training #####

def train_yolo(request):
    if request.method == 'POST':
        try:
            # Handle the uploaded YAML file
            yaml_file = request.FILES.get('yaml_file')
            if not yaml_file:
                return JsonResponse({'status': 'error', 'message': 'No YAML file uploaded'})

            # Get dataset path from user input
            dataset_path = request.POST.get('dataset_path')
            if not dataset_path:
                return JsonResponse({'status': 'error', 'message': 'No dataset path provided'})

            # Save the uploaded YAML file
            upload_dir = 'C:\\Users\\asad3\\pop\\model_results'  # Change this to your desired upload directory
            os.makedirs(upload_dir, exist_ok=True)
            yaml_path = os.path.join(upload_dir, yaml_file.name)
            with open(yaml_path, 'wb') as f:
                for chunk in yaml_file.chunks():
                    f.write(chunk)

            # Read the YAML file and update paths
            with open(yaml_path, 'r') as f:
                yaml_content = f.read()
            updated_yaml_content = yaml_content.replace('../train/images', f'{dataset_path}/train/images')
            updated_yaml_content = updated_yaml_content.replace('../valid/images', f'{dataset_path}/valid/images')
            updated_yaml_content = updated_yaml_content.replace('../test/images', f'{dataset_path}/test/images')
            with open(yaml_path, 'w') as f:
                f.write(updated_yaml_content)

            # Rest of your YOLOv8 training code goes here...
            model = YOLO('yolov8n.pt')

            # Training using the updated data.yaml file
            results = model.train(
                data=yaml_path,
                epochs=30,
                imgsz=256
            )

            # Save results to directory
            output_dir = "model_results"
            os.makedirs(output_dir, exist_ok=True)
            results.save(output_dir)

            response_data = {
                'status': 'success',
                'message': 'Training completed successfully',
                'results_dir': output_dir
            }

            return JsonResponse(response_data)

        except Exception as e:
            error_response = {
                'status': 'error',
                'message': str(e)
            }
            return JsonResponse(error_response)
    else:
        return JsonResponse({'status': 'error', 'message': 'Invalid request method'})




################################################################################################################################

                                                ##### Anomaly detection Inference #####

import os
import numpy as np
from PIL import Image
import tempfile
import base64
import io
import numpy as np
import io
import base64
from PIL import Image
import io
import torch

def anomaly_detection_api(request):
    if request.method == 'POST':
        try:
            image_data = request.FILES.get('image')

            if image_data is None:
                return JsonResponse({'error': 'No image data provided'}, status=400)

            model_path = request.POST.get('model_path')


            if model_path is None:
                return JsonResponse({'error': 'Model path path not provided'}, status=400)

            image_bytes = image_data.read()
            image_array = np.frombuffer(image_bytes, np.uint8)
            image_path = os.path.join(tempfile.gettempdir(), "temp_image.jpg")
    
            with open(image_path, "wb") as img_file:
                img_file.write(image_array)

            output='C:\\Users\\asad3\\Documents\\xis\\usaisoft-platform-backend\\ok\\results'
            train_args = Namespace(
                weights=model_path,
                input=image_path,
                output=output,
                task='segmentation',
                visualization_mode='simple',
                device='auto',
                log_level="INFO"
            )
            results = infer(train_args)
            
            image = Image.fromarray(results.astype('uint8'))
            image_buffer = io.BytesIO()
            image.save(image_buffer, format="PNG")
            image_buffer.seek(0)
            response = HttpResponse(image_buffer.read(), content_type="image/png")
            return response 
    
        except Exception as e:
            error_message = str(e)
            return JsonResponse({'error': error_message}, status=500)

    return JsonResponse({'error': 'Invalid request method'}, status=405)



##############################################################################################################################

                                   ##### Anomaly Detection Training #####


def train_anomaly_detection(request):
    
        if request.method == 'POST':
            try:
                config = request.POST.get('config_path')
                if config is None:
                    return JsonResponse({'error': 'Model path or config path not provided'}, status=400)
        # Set the model and config paths here
                model = "cfs"
        #config = "C:\\Users\\asad3\\Documents\\xis\\usaisoft-platform-backend\\ok\\config.yaml"

        # Create the arguments Namespace
                train_args = Namespace(model=model, config=config, log_level="INFO")

        # Call the train function from the train.py script
                train(train_args)

                return JsonResponse({'status': 'Model training process initiated!'})
            except Exception as e:
        # Handle any errors that occur during training
                return JsonResponse({'error': str(e)}, status=500)
    



##############################################################################################################################

                                         ##### Creating YAML File #####


def create_subfolders(base_dir):
    os.makedirs(os.path.join(base_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'labels'), exist_ok=True)

def move_files(files, source_dir, destination_dir):
    for file in files:
        image_path = os.path.join(source_dir, 'images', file)
        label_file = file.replace(".jpg", ".txt")
        label_path = os.path.join(source_dir, 'labels', label_file)

        shutil.move(image_path, os.path.join(destination_dir, 'images', file))
        shutil.move(label_path, os.path.join(destination_dir, 'labels', label_file))

def remove_original_folders(directory):
    images_dir = os.path.join(directory, 'images')
    labels_dir = os.path.join(directory, 'labels')
    shutil.rmtree(images_dir)
    shutil.rmtree(labels_dir)

def split_data(src_dir, train_ratio=0.6, valid_ratio=0.3, test_ratio=0.1):
    # Create target directories if they don't exist
    train_dir = os.path.join(src_dir, 'train')
    valid_dir = os.path.join(src_dir, 'valid')
    test_dir = os.path.join(src_dir, 'test')

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(valid_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    images_dir = os.path.join(src_dir, 'images')
    labels_dir = os.path.join(src_dir, 'labels')

    # Get the list of image files in the images folder
    image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
    num_images = len(image_files)

    # Shuffle the files randomly
    random.shuffle(image_files)

    # Calculate the number of files for each split
    num_train = int(num_images * train_ratio)
    num_valid = int(num_images * valid_ratio)

    # Split the files into train, valid, and test sets
    train_files = image_files[:num_train]
    valid_files = image_files[num_train:num_train + num_valid]
    test_files = image_files[num_train + num_valid:]

    create_subfolders(train_dir)
    create_subfolders(valid_dir)
    create_subfolders(test_dir)

    move_files(train_files, src_dir, train_dir)
    move_files(valid_files, src_dir, valid_dir)
    move_files(test_files, src_dir, test_dir)

    # Remove the original "images" and "labels" folders
    remove_original_folders(src_dir)


def create_yaml_file(dirr):
    classes_txt_file_path = os.path.join(dirr, 'classes.txt')
    with open(classes_txt_file_path, 'r') as txt_file:
        class_names = [line.strip() for line in txt_file.readlines()]

    yaml_data = {
        'train': '../train/images',
        'val': '../valid/images',
        'test': '../test/images',
        'nc': len(class_names),
        'names': class_names
    }

    yaml_file_path = os.path.join(dirr, 'data.yaml')
    with open(yaml_file_path, 'w') as yaml_file:
        yaml_file.write("train: {}\n".format(yaml_data['train']))
        yaml_file.write("val: {}\n".format(yaml_data['val']))
        yaml_file.write("test: {}\n".format(yaml_data['test']))
        yaml_file.write("\n")
        yaml_file.write("nc: {}\n".format(yaml_data['nc']))
        yaml_file.write("names: {}\n".format(str(yaml_data['names'])))


def process_directory(request):
    if request.method == 'POST':
        directory_path = request.POST.get('directory_path')

        if directory_path:
            try:
                split_data(directory_path)
                create_yaml_file(directory_path)
                return JsonResponse({'status': 'success', 'message': 'Directory processed successfully.'})
            except Exception as e:
                return JsonResponse({'status': 'error', 'message': str(e)})
        else:
            return JsonResponse({'status': 'error', 'message': 'Directory path not provided.'})
    else:
        return JsonResponse({'status': 'error', 'message': 'Invalid request method.'})
    
##############################################################################################################################



def home(request):
  

    return HttpResponse("Home function executed successfully!")


############################################################################################################################

# myapp/views.py

from django.http import HttpResponse
from django.views.decorators.http import require_http_methods
import cv2
import time
from pyueye import ueye

model = YOLO("yolov8n.pt")

def webcam(request):
    return render(request, 'webcam_app/webcam.html')

def video_feed(request):
    #if request.method == 'POST' or request.method == 'GET':
        def generate_frames():

            hCam = ueye.HIDS(0)             #0: first available camera;  1-254: The camera with the specified camera ID
            sInfo = ueye.SENSORINFO()
            cInfo = ueye.CAMINFO()
            pcImageMemory = ueye.c_mem_p()
            MemID = ueye.int()
            rectAOI = ueye.IS_RECT()
            pitch = ueye.INT()
            nBitsPerPixel = ueye.INT(24)    #24: bits per pixel for color mode; take 8 bits per pixel for monochrome
            channels = 3                    #3: channels for color mode(RGB); take 1 channel for monochrome
            m_nColorMode = ueye.INT()		# Y8/RGB16/RGB24/REG32
            bytes_per_pixel = int(nBitsPerPixel / 8)

            nRet = ueye.is_InitCamera(hCam, None)
            if nRet != ueye.IS_SUCCESS:
                print("is_InitCamera ERROR")

            nRet = ueye.is_GetCameraInfo(hCam, cInfo)
            if nRet != ueye.IS_SUCCESS:
                print("is_GetCameraInfo ERROR")

            nRet = ueye.is_GetSensorInfo(hCam, sInfo)
            if nRet != ueye.IS_SUCCESS:
                print("is_GetSensorInfo ERROR")

            nRet = ueye.is_ResetToDefault( hCam)
            if nRet != ueye.IS_SUCCESS:
                print("is_ResetToDefault ERROR")

            nRet = ueye.is_SetDisplayMode(hCam, ueye.IS_SET_DM_DIB)

            if int.from_bytes(sInfo.nColorMode.value, byteorder='big') == ueye.IS_COLORMODE_BAYER:
            # setup the color depth to the current windows setting
                ueye.is_GetColorDepth(hCam, nBitsPerPixel, m_nColorMode)
                bytes_per_pixel = int(nBitsPerPixel / 8)
                print("IS_COLORMODE_BAYER: ", )
                print("\tm_nColorMode: \t\t", m_nColorMode)
                print("\tnBitsPerPixel: \t\t", nBitsPerPixel)
                print("\tbytes_per_pixel: \t\t", bytes_per_pixel)
                print()

            elif int.from_bytes(sInfo.nColorMode.value, byteorder='big') == ueye.IS_COLORMODE_CBYCRY:
                # for color camera models use RGB32 mode
                m_nColorMode = ueye.IS_CM_BGRA8_PACKED
                nBitsPerPixel = ueye.INT(32)
                bytes_per_pixel = int(nBitsPerPixel / 8)
                print("IS_COLORMODE_CBYCRY: ", )
                print("\tm_nColorMode: \t\t", m_nColorMode)
                print("\tnBitsPerPixel: \t\t", nBitsPerPixel)
                print("\tbytes_per_pixel: \t\t", bytes_per_pixel)
                print()

            elif int.from_bytes(sInfo.nColorMode.value, byteorder='big') == ueye.IS_COLORMODE_MONOCHROME:
                # for color camera models use RGB32 mode
                m_nColorMode = ueye.IS_CM_MONO8
                nBitsPerPixel = ueye.INT(8)
                bytes_per_pixel = int(nBitsPerPixel / 8)
                print("IS_COLORMODE_MONOCHROME: ", )
                print("\tm_nColorMode: \t\t", m_nColorMode)
                print("\tnBitsPerPixel: \t\t", nBitsPerPixel)
                print("\tbytes_per_pixel: \t\t", bytes_per_pixel)
                print()

            
            else:
            # for monochrome camera models use Y8 mode
                m_nColorMode = ueye.IS_CM_MONO8
                nBitsPerPixel = ueye.INT(8)
                bytes_per_pixel = int(nBitsPerPixel / 8)
                print("else")
            
            nRet = ueye.is_AOI(hCam, ueye.IS_AOI_IMAGE_GET_AOI, rectAOI, ueye.sizeof(rectAOI))
            if nRet != ueye.IS_SUCCESS:
                print("is_AOI ERROR")

            width = rectAOI.s32Width
            height = rectAOI.s32Height

            print("Camera model:\t\t", sInfo.strSensorName.decode('utf-8'))
            print("Camera serial no.:\t", cInfo.SerNo.decode('utf-8'))
            print("Maximum image width:\t", width)
            print("Maximum image height:\t", height)
            print()

            nRet = ueye.is_AllocImageMem(hCam, width, height, nBitsPerPixel, pcImageMemory, MemID)
            if nRet != ueye.IS_SUCCESS:
                print("is_AllocImageMem ERROR")
            else:
                # Makes the specified image memory the active memory
                nRet = ueye.is_SetImageMem(hCam, pcImageMemory, MemID)
                if nRet != ueye.IS_SUCCESS:
                    print("is_SetImageMem ERROR")
                else:
                    # Set the desired color mode
                    nRet = ueye.is_SetColorMode(hCam, m_nColorMode)

            # Activates the camera's live video mode (free run mode)
            nRet = ueye.is_CaptureVideo(hCam, ueye.IS_DONT_WAIT)
            if nRet != ueye.IS_SUCCESS:
                print("is_CaptureVideo ERROR")

            # Enables the queue mode for existing image memory sequences
            nRet = ueye.is_InquireImageMem(hCam, pcImageMemory, MemID, width, height, nBitsPerPixel, pitch)
            if nRet != ueye.IS_SUCCESS:
                print("is_InquireImageMem ERROR")
            else:
                print("Press q to leave the programm")

            

            while(nRet == ueye.IS_SUCCESS):

                array = ueye.get_data(pcImageMemory, width, height, nBitsPerPixel, pitch, copy=False)
                frame = np.reshape(array,(height.value, width.value, bytes_per_pixel))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame,(0,0),fx=0.5, fy=0.5)
                results = model(frame)

                annotated_frame = results[0].plot()  # Use result[0].plot() to annotate the frame

                _, buffer = cv2.imencode('.jpg', annotated_frame)

                # Convert frame to bytes for streaming
                frame_bytes = buffer.tobytes()
                
                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                

        return StreamingHttpResponse(generate_frames(), content_type='multipart/x-mixed-replace; boundary=frame')
    #return HttpResponse('Invalid request method', status=405)


from anomalib.deploy import TorchInferencer
from anomalib.post_processing import Visualizer


def webcam(request):
    return render(request, 'webcam_app/anomalib_webcam.html')

def anomalib_cam(request):
        inferencer = TorchInferencer(path='C:\\Users\\asad3\\Documents\\xis\\usaisoft-platform-backend\\results\\cfa\\Paper\\run\\weights\\torch\\model.pt', device='auto')
        visualizer = Visualizer(mode='simple', task='segmentation')
    #if request.method == 'POST' or request.method == 'GET':
        def generate_frames():

            hCam = ueye.HIDS(0)             #0: first available camera;  1-254: The camera with the specified camera ID
            sInfo = ueye.SENSORINFO()
            cInfo = ueye.CAMINFO()
            pcImageMemory = ueye.c_mem_p()
            MemID = ueye.int()
            rectAOI = ueye.IS_RECT()
            pitch = ueye.INT()
            nBitsPerPixel = ueye.INT(24)    #24: bits per pixel for color mode; take 8 bits per pixel for monochrome
            channels = 3                    #3: channels for color mode(RGB); take 1 channel for monochrome
            m_nColorMode = ueye.INT()		# Y8/RGB16/RGB24/REG32
            bytes_per_pixel = int(nBitsPerPixel / 8)

            nRet = ueye.is_InitCamera(hCam, None)
            if nRet != ueye.IS_SUCCESS:
                print("is_InitCamera ERROR")

            nRet = ueye.is_GetCameraInfo(hCam, cInfo)
            if nRet != ueye.IS_SUCCESS:
                print("is_GetCameraInfo ERROR")

            nRet = ueye.is_GetSensorInfo(hCam, sInfo)
            if nRet != ueye.IS_SUCCESS:
                print("is_GetSensorInfo ERROR")

            nRet = ueye.is_ResetToDefault( hCam)
            if nRet != ueye.IS_SUCCESS:
                print("is_ResetToDefault ERROR")

            nRet = ueye.is_SetDisplayMode(hCam, ueye.IS_SET_DM_DIB)

            if int.from_bytes(sInfo.nColorMode.value, byteorder='big') == ueye.IS_COLORMODE_BAYER:
            # setup the color depth to the current windows setting
                ueye.is_GetColorDepth(hCam, nBitsPerPixel, m_nColorMode)
                bytes_per_pixel = int(nBitsPerPixel / 8)
                print("IS_COLORMODE_BAYER: ", )
                print("\tm_nColorMode: \t\t", m_nColorMode)
                print("\tnBitsPerPixel: \t\t", nBitsPerPixel)
                print("\tbytes_per_pixel: \t\t", bytes_per_pixel)
                print()

            elif int.from_bytes(sInfo.nColorMode.value, byteorder='big') == ueye.IS_COLORMODE_CBYCRY:
                # for color camera models use RGB32 mode
                m_nColorMode = ueye.IS_CM_BGRA8_PACKED
                nBitsPerPixel = ueye.INT(32)
                bytes_per_pixel = int(nBitsPerPixel / 8)
                print("IS_COLORMODE_CBYCRY: ", )
                print("\tm_nColorMode: \t\t", m_nColorMode)
                print("\tnBitsPerPixel: \t\t", nBitsPerPixel)
                print("\tbytes_per_pixel: \t\t", bytes_per_pixel)
                print()

            elif int.from_bytes(sInfo.nColorMode.value, byteorder='big') == ueye.IS_COLORMODE_MONOCHROME:
                # for color camera models use RGB32 mode
                m_nColorMode = ueye.IS_CM_MONO8
                nBitsPerPixel = ueye.INT(8)
                bytes_per_pixel = int(nBitsPerPixel / 8)
                print("IS_COLORMODE_MONOCHROME: ", )
                print("\tm_nColorMode: \t\t", m_nColorMode)
                print("\tnBitsPerPixel: \t\t", nBitsPerPixel)
                print("\tbytes_per_pixel: \t\t", bytes_per_pixel)
                print()

            
            else:
            # for monochrome camera models use Y8 mode
                m_nColorMode = ueye.IS_CM_MONO8
                nBitsPerPixel = ueye.INT(8)
                bytes_per_pixel = int(nBitsPerPixel / 8)
                print("else")
            
            nRet = ueye.is_AOI(hCam, ueye.IS_AOI_IMAGE_GET_AOI, rectAOI, ueye.sizeof(rectAOI))
            if nRet != ueye.IS_SUCCESS:
                print("is_AOI ERROR")

            width = rectAOI.s32Width
            height = rectAOI.s32Height

            print("Camera model:\t\t", sInfo.strSensorName.decode('utf-8'))
            print("Camera serial no.:\t", cInfo.SerNo.decode('utf-8'))
            print("Maximum image width:\t", width)
            print("Maximum image height:\t", height)
            print()

            nRet = ueye.is_AllocImageMem(hCam, width, height, nBitsPerPixel, pcImageMemory, MemID)
            if nRet != ueye.IS_SUCCESS:
                print("is_AllocImageMem ERROR")
            else:
                # Makes the specified image memory the active memory
                nRet = ueye.is_SetImageMem(hCam, pcImageMemory, MemID)
                if nRet != ueye.IS_SUCCESS:
                    print("is_SetImageMem ERROR")
                else:
                    # Set the desired color mode
                    nRet = ueye.is_SetColorMode(hCam, m_nColorMode)

            # Activates the camera's live video mode (free run mode)
            nRet = ueye.is_CaptureVideo(hCam, ueye.IS_DONT_WAIT)
            if nRet != ueye.IS_SUCCESS:
                print("is_CaptureVideo ERROR")

            # Enables the queue mode for existing image memory sequences
            nRet = ueye.is_InquireImageMem(hCam, pcImageMemory, MemID, width, height, nBitsPerPixel, pitch)
            if nRet != ueye.IS_SUCCESS:
                print("is_InquireImageMem ERROR")
            else:
                print("Press q to leave the programm")

            

            while(nRet == ueye.IS_SUCCESS):

                array = ueye.get_data(pcImageMemory, width, height, nBitsPerPixel, pitch, copy=False)
                frame = np.reshape(array,(height.value, width.value, bytes_per_pixel))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame,(0,0),fx=0.5, fy=0.5)

                predictions = inferencer.predict(frame)
                annotated_frame = visualizer.visualize_image(predictions) 
                _, buffer = cv2.imencode('.jpg', annotated_frame)

                frame_bytes = buffer.tobytes()
                
                
                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        


        return StreamingHttpResponse(generate_frames(), content_type='multipart/x-mixed-replace; boundary=frame')