import datetime
import io
import json
import cv2
import time
import os
import base64
import shutil
import random
import yaml
import tempfile
import numpy as np
import subprocess
from PIL import Image
from pyueye import ueye
from ultralytics import YOLO
from datetime import datetime
from django.shortcuts import render
from django.http import HttpResponseBadRequest, JsonResponse
from .torch_inference import infer
from ultralytics.utils import ROOT, yaml_load
from ultralytics.utils.checks import check_yaml
from .anomaly_training import train
from argparse import Namespace
from anomalib.deploy import TorchInferencer
from anomalib.post_processing import Visualizer
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.core.serializers.json import DjangoJSONEncoder
from django.http import StreamingHttpResponse, HttpResponseServerError






def yolo(image_bytes, model_path):
    try:
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
        if 'model_path' in request.FILES and 'image' in request.FILES:
            model_path = request.FILES['model_path']
            image = request.FILES['image'].read()

            current_directory = os.getcwd()
            model_filename = model_path.name
            model_path_asd = os.path.join(current_directory, model_filename)
            with open(model_path_asd, 'wb') as destination:
                for chunk in model_path.chunks():
                    destination.write(chunk)
            print(f'Model file received and saved at: {model_path_asd}')

            try:
                detections = yolo(image, model_path_asd)
                temp_output_path = os.path.join(tempfile.gettempdir(), 'output.jpg')
                cv2.imwrite(temp_output_path, detections)

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



def train_yolo(request):
    if request.method == 'POST':
        try:
            yaml_file = request.FILES.get('yaml_file')
            if not yaml_file:
                return JsonResponse({'status': 'error', 'message': 'No YAML file uploaded'})
            dataset_path = request.POST.get('dataset_path')
            if not dataset_path:
                return JsonResponse({'status': 'error', 'message': 'No dataset path provided'})
            training_speed = request.POST.get('training_speed')
            print("Selected Training Speed:", training_speed)
            upload_dir = './model_results'  
            os.makedirs(upload_dir, exist_ok=True)
            yaml_path = os.path.join(upload_dir, yaml_file.name)
            with open(yaml_path, 'wb') as f:
                for chunk in yaml_file.chunks():
                    f.write(chunk)

            with open(yaml_path, 'r') as f:
                yaml_content = f.read()
            updated_yaml_content = yaml_content.replace('../train/images', f'{dataset_path}/train/images')
            updated_yaml_content = updated_yaml_content.replace('../valid/images', f'{dataset_path}/valid/images')
            updated_yaml_content = updated_yaml_content.replace('../test/images', f'{dataset_path}/test/images')
            with open(yaml_path, 'w') as f:
                f.write(updated_yaml_content)

            if training_speed =='fast':
                model = YOLO('yolov8n.pt')

            
                results = model.train(
                    data=yaml_path,
                    epochs=50,
                    #patience=6,
                    imgsz=256
                )
                output_dir = "model_results"
                os.makedirs(output_dir, exist_ok=True)
                results.save(output_dir)
                response_data = {
                    'status': 'success',
                    'message': 'Training completed successfully',
                    'results_dir': output_dir
                }

                return JsonResponse(response_data)
                
            else:
                model = YOLO('yolov8m.pt')

            
                results = model.train(
                    data=yaml_path,
                    epochs=100,
                    #patience=6,
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






def anomaly_detection_api(request):
    if request.method == 'POST':
        try:
            image_data = request.FILES.get('image')

            if image_data is None:
                return JsonResponse({'error': 'No image data provided'}, status=400)

            model_path = request.FILES.get('model_path')
            print(model_path)

            if model_path is None:
                return JsonResponse({'error': 'Model path path not provided'}, status=400)
            

            current_directory = os.getcwd()
            model_filename = model_path.name
            model_path_asd = os.path.join(current_directory, model_filename)
            with open(model_path_asd, 'wb') as destination:
                for chunk in model_path.chunks():
                    destination.write(chunk)
            print(f'Model file received and saved at: {model_path_asd}')

            image_bytes = image_data.read()
            image_array = np.frombuffer(image_bytes, np.uint8)
            image_path = os.path.join(tempfile.gettempdir(), "temp_image.jpg")
    
            with open(image_path, "wb") as img_file:
                img_file.write(image_array)
            output='C:\\Users\\asad3\\Documents\\xis\\usaisoft-platform-backend\\ok\\results'
            train_args = Namespace(
                weights=model_path_asd,
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







def train_anomaly_detection(request):
    
    if request.method == 'POST':
        try:
            config_upload = request.FILES.get('config_path')
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                for chunk in config_upload.chunks():
                    temp_file.write(chunk)
    
            temp_file_path = temp_file.name
            with open(temp_file_path, 'r') as config_file:
                config = yaml.load(config_file, Loader=yaml.FullLoader)
            
            model_name = str(config.get('model', {}).get('name'))
            print(model_name)
            train_args = Namespace(model=model_name, config=temp_file_path, log_level="INFO")
            train(train_args)
            os.remove(temp_file_path)
            
            return JsonResponse({'status': 'Model training process initiated!'})
        
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    

              


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
    




def home(request):
  
    return HttpResponse("Home function executed successfully!")



@require_http_methods(["GET", "POST"])
def model_upload_yolo(request):
    try:
        if request.method == 'POST':
            model_file = request.FILES.get('model_file')
            if model_file is None:
                return HttpResponseBadRequest("No model file uploaded.")
            
            model_filename = 'yolo.pt'
            model_path = os.path.join(os.getcwd(), model_filename)

            with open(model_path, 'wb') as destination:
                for chunk in model_file.chunks():
                    destination.write(chunk)

            print(f'Model file received and saved at: {model_path}')
            return HttpResponse("Model file uploaded successfully.")
        return HttpResponse()
    except Exception as e:
        print(str(e))  
        return HttpResponseServerError("Internal Server Error")


@require_http_methods(["GET", "POST"])
def yolo_webcam(request):
    try:
        model = YOLO('./yolo.pt')  
        def generate_frames():
            cap = cv2.VideoCapture(0)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if model is not None:  
                    results = model(frame)
                    annotated_frame = results[0].plot()
                    _, buffer = cv2.imencode('.jpg', annotated_frame)
                    frame = buffer.tobytes()

                    yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
                

        response = StreamingHttpResponse(
            generate_frames(),
            content_type="multipart/x-mixed-replace; boundary=frame",
        )
        return response
    except Exception as e:
        print(str(e))  
        return HttpResponseServerError("Internal Server Error")

    #return render(request, 'Yolo_live_inference.html')




@require_http_methods(["GET", "POST"])
def yolo_industrial(request):
    try:
        model = YOLO('./yolo.pt')  
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

                annotated_frame = results[0].plot()  

                _, buffer = cv2.imencode('.jpg', annotated_frame)

                # Convert frame to bytes for streaming
                frame_bytes = buffer.tobytes()
                
                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        response = StreamingHttpResponse(
            generate_frames(),
            content_type="multipart/x-mixed-replace; boundary=frame",
        )
        return response
    except Exception as e:
        print(str(e))  
        return HttpResponseServerError("Internal Server Error")




@require_http_methods(["GET", "POST"])
def model_upload_anomalib(request):
    try:
        if request.method == 'POST':
            model_file = request.FILES.get('model_file')
            if model_file is None:
                return HttpResponseBadRequest("No model file uploaded.")
            
            model_filename = 'anomalib.pt'
            model_path = os.path.join(os.getcwd(), model_filename)

            with open(model_path, 'wb') as destination:
                for chunk in model_file.chunks():
                    destination.write(chunk)

            print(f'Model file received and saved at: {model_path}')
            return HttpResponse("Model file uploaded successfully.")
        return HttpResponse()
    except Exception as e:
        print(str(e))  
        return HttpResponseServerError("Internal Server Error")


@require_http_methods(["GET", "POST"])
def anomalib_webcam(request):
    try:
        model_asd='anomalib.pt'
        
        inferencer = TorchInferencer(path=model_asd, device='auto')
        visualizer = Visualizer(mode='simple', task='segmentation')
            
        def generate_frames():
            cap = cv2.VideoCapture(0)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                predictions = inferencer.predict(frame)
                annotated_frame = visualizer.visualize_image(predictions)
                _, buffer = cv2.imencode('.jpg', annotated_frame)
                frame = buffer.tobytes()

                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
                time.sleep(0.1)
        response = StreamingHttpResponse(
            generate_frames(),
            content_type="multipart/x-mixed-replace; boundary=frame",
        )

        return response
    except Exception as e:
        print(str(e))
        return HttpResponseServerError("Internal Server Error")
    
    #return render(request, 'Anomalib_live_inference.html')


@require_http_methods(["GET", "POST"])
def anomalib_industrial(request):
    try:
        model_asd='anomalib.pt'
        
        inferencer = TorchInferencer(path=model_asd, device='auto')
        visualizer = Visualizer(mode='simple', task='segmentation')
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
        response = StreamingHttpResponse(
            generate_frames(),
            content_type="multipart/x-mixed-replace; boundary=frame",
        )

        return response
    except Exception as e:
        print(str(e))
        return HttpResponseServerError("Internal Server Error")



frame_number = 0  
capture_frame = False 
base_directory = "./Data capture"
timestamp = datetime.now().strftime("Webcam_%Y%m%d%H%M%S")
folder_name = f"{timestamp}"
frame_directory = os.path.join(base_directory, folder_name)
os.makedirs(frame_directory) 

@csrf_exempt
def webcam(request): 
    
    def generate_frames():
        global frame_number, capture_frame
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            global frame_number, capture_frame
            if capture_frame:
                frame_number += 1
                frame_path = os.path.join(frame_directory, f'frame_{frame_number}.jpg')
                cv2.imwrite(frame_path, frame)
                capture_frame = False

            _, buffer = cv2.imencode('.jpg', frame)
            frame_data = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')

    if request.method == 'POST':
        global capture_frame
        capture_frame = True
        return JsonResponse({'message': 'Frame capture request received'})

    response = StreamingHttpResponse(generate_frames(), content_type="multipart/x-mixed-replace;boundary=frame")
    return response



frame_number = 0  
capture_frame = False 
base_directory = "./Data capture"
timestamp = datetime.now().strftime("Industrial_%Y%m%d%H%M%S")
folder_name = f"{timestamp}"
frame_directory1 = os.path.join(base_directory, folder_name)
os.makedirs(frame_directory1)

@csrf_exempt
def webcam_industrial(request): 
    
    def generate_frames():
            
            global frame_number, capture_frame
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
                global frame_number, capture_frame
                if capture_frame:
                    
                    frame_number += 1
                    frame_path = os.path.join(frame_directory1, f'frame_{frame_number}.jpg')
                    cv2.imwrite(frame_path, frame)
                    capture_frame = False

                _, buffer = cv2.imencode('.jpg', frame)
                frame_bytes = buffer.tobytes()
                
                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                
            if request.method == 'POST':
                global capture_frame
                capture_frame = True
                return JsonResponse({'message': 'Frame capture request received'})

            response = StreamingHttpResponse(generate_frames(), content_type="multipart/x-mixed-replace;boundary=frame")
            return response
                

