import datetime
import io
import json
import cv2
import time
import os
import shutil
import random
import yaml
import tempfile
import pandas as pd
from typing import List
import numpy as np
from PIL import Image
from ultralytics import YOLO
from datetime import datetime
from django.shortcuts import render
from django.http import HttpResponseBadRequest, JsonResponse
from .torch_inference import infer
from ultralytics.utils import ROOT
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



def train_yolo(request):
    if request.method == 'POST':
            zip_file = request.FILES.get('zip_file')

            # Create a directory for permanent storage if not exists
            storage_dir = os.path.join(os.getcwd(), 'Yolo Training Dataset')
            os.makedirs(storage_dir, exist_ok=True)

            # Save the uploaded ZIP file permanently
            zip_path = os.path.join(storage_dir, zip_file.name)
            with open(zip_path, 'wb') as f:
                for chunk in zip_file.chunks():
                    f.write(chunk)

            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(storage_dir)

            # Find the YAML file within the extracted contents
            yaml_file = None
            for root, dirs, files in os.walk(storage_dir):
                for file in files:
                    if file.endswith('.yaml'):
                        yaml_file = os.path.join(root, file)
                        break
                if yaml_file:
                    break

            if not yaml_file:
                return JsonResponse({'status': 'error', 'message': 'No YAML file found in the ZIP archive'})

            training_speed = request.POST.get('training_speed')
            print("Selected Training Speed:", training_speed)
            yaml_path = os.path.join(storage_dir, os.path.basename(yaml_file))

            with open(yaml_file, 'r') as source:
                source_content = source.read()

            with open(yaml_path, 'w') as destination:
                destination.write(source_content)

            if training_speed == 'fast':
                model = YOLO('yolov8n.pt')
                epochs = 50
            else:
                model = YOLO('yolov8m.pt')
                epochs = 100

            # Training using the updated data.yaml file
            results = model.train(
                data=yaml_path,
                epochs=epochs,
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

    else:
        return JsonResponse({'status': 'error', 'message': 'Invalid request method'})






def anomaly_detection_api(request):
    if request.method == 'POST':
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
    

    return JsonResponse({'error': 'Invalid request method'}, status=405)







def train_anomaly_detection(request):
    
    if request.method == 'POST':
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
    supported_extensions = (".png", ".jpg", ".jpeg", ".webp", ".svg")
    image_files = [f for f in os.listdir(images_dir) if f.endswith(supported_extensions)]
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






from django.core.files.storage import FileSystemStorage
import zipfile
import time
import uuid

def anomalib_config(request):
    if request.method == 'POST':
        # Generate a unique folder name
        timestamp = int(time.time())
        unique_id = str(uuid.uuid4())[:8]
        folder_name = f"anomalib_data_{timestamp}_{unique_id}"

        # Create a folder for anomlib data inside the "Dataset" directory
        dataset_folder = os.path.join("Datasets", folder_name)
        if not os.path.exists(dataset_folder):
            os.mkdir(dataset_folder)

        # Extract and save the normal images
        normal_zip = request.FILES['normal_zip']
        normal_dir = os.path.join(dataset_folder, "good")
        with zipfile.ZipFile(normal_zip, 'r') as zip_ref:
            zip_ref.extractall(normal_dir)

        # Extract and save the abnormal images
        abnormal_zip = request.FILES['abnormal_zip']
        abnormal_dir = os.path.join(dataset_folder, "colour")
        with zipfile.ZipFile(abnormal_zip, 'r') as zip_ref:
            zip_ref.extractall(abnormal_dir)
        rename_files(abnormal_dir)

        # Extract and save the mask images
        mask_zip = request.FILES['mask_zip']
        mask_dir = os.path.join(dataset_folder, "mask")
        with zipfile.ZipFile(mask_zip, 'r') as zip_ref:
            zip_ref.extractall(mask_dir)
        rename_files(mask_dir)

        config_file_path = "./config.yaml"
        with open(config_file_path, 'r') as config_file:
            config = yaml.safe_load(config_file)

        # Update the root path to the current folder
        config['dataset']['root'] = os.path.abspath(dataset_folder)

        with open(config_file_path, 'w') as config_file:
            yaml.dump(config, config_file, default_flow_style=False)

        return HttpResponse(f"Config file generated : {folder_name}")
    else:
        return HttpResponse("Invalid request method.")

def rename_files(directory):
    files = os.listdir(directory)
    for i, file in enumerate(files):
        file_extension = os.path.splitext(file)[1]
        new_name = str(i + 1) + file_extension
        os.rename(os.path.join(directory, file), os.path.join(directory, new_name))

def create_subdirectories_and_move_files(src_dir, dest_dir, file_list, subdir_name):
    try:
        os.makedirs(os.path.join(dest_dir, subdir_name), exist_ok=True)

        for file in file_list:
            src_file = os.path.join(src_dir, file)
            dest_file = os.path.join(dest_dir, subdir_name, file)
            shutil.move(src_file, dest_file)
    except Exception as e:
        print(f"Error during subdirectory creation and file movement: {e}")

def split_data(main_dir, images_dir, labels_dir, split_ratio=[0.8, 0.1, 0.1]):
    try:
        os.makedirs(main_dir, exist_ok=True)

        train_dir = os.path.join(main_dir, 'train')
        test_dir = os.path.join(main_dir, 'test')
        val_dir = os.path.join(main_dir, 'val')
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)

        # Get a list of image and label files
        supported_extensions = (".png", ".jpg", ".jpeg", ".webp", ".svg")
        image_files = [f for f in os.listdir(images_dir) if f.endswith(supported_extensions)]
        label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]

        num_files = len(image_files)

        if num_files != len(label_files):
            print("Mismatch between image and label files")

        num_train = int(split_ratio[0] * num_files)
        num_test = int(split_ratio[1] * num_files)
        num_val = num_files - num_train - num_test

        random.shuffle(image_files)

        # Distribute the images
        create_subdirectories_and_move_files(images_dir, train_dir, image_files[:num_train], "images")
        create_subdirectories_and_move_files(images_dir, test_dir, image_files[num_train:num_train + num_test], "images")
        create_subdirectories_and_move_files(images_dir, val_dir, image_files[num_train + num_test:], "images")

        # Copy the corresponding labels to the same directories
        create_subdirectories_and_move_files(labels_dir, train_dir, [f"{img.split('.')[0]}.txt" for img in image_files[:num_train]], "labels")
        create_subdirectories_and_move_files(labels_dir, test_dir, [f"{img.split('.')[0]}.txt" for img in image_files[num_train:num_train + num_test]], "labels")
        create_subdirectories_and_move_files(labels_dir, val_dir, [f"{img.split('.')[0]}.txt" for img in image_files[num_train + num_test:]], "labels")

        # Remove the original image and label folders
        shutil.rmtree(images_dir)
        shutil.rmtree(labels_dir)
    except Exception as e:
        print(f"Error during data splitting: {e}")

def yolo_csv(request):
    if request.method == 'POST' and request.FILES.get('csv_file'):
        csv_file = request.FILES['csv_file']

        df = pd.read_csv(csv_file)

        main_dir = 'Yolo Dataset'
        os.makedirs(main_dir, exist_ok=True)

        # Create images and labels directories
        images_dir = os.path.join(main_dir, 'images')
        labels_dir = os.path.join(main_dir, 'labels')
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)

        image_path = df['image'][0]
        folder_name = os.path.dirname(image_path).split('/')[-1]
        combined_path = os.path.join("C:\\Users\\asad3\\AppData\\Local\\label-studio\\label-studio\\media\\upload", folder_name)
        print(combined_path)

        class InputStream:
            def __init__(self, data):
                self.data = data
                self.i = 0

            def read(self, size):
                out = self.data[self.i:self.i + size]
                self.i += size
                return int(out, 2)

        def access_bit(data, num):
            """ from bytes array to bits by num position"""
            base = int(num // 8)
            shift = 7 - int(num % 8)
            return (data[base] & (1 << shift)) >> shift

        def bytes2bit(data):
            """ get bit string from bytes data"""
            return ''.join([str(access_bit(data, i)) for i in range(len(data) * 8)])

        def rle_to_mask(rle: List[int], height: int, width: int) -> np.array:
            """
            Converts rle to image mask
            Args:
                rle: your long rle
                height: original_height
                width: original_width

            Returns: np.array
            """

            rle_input = InputStream(bytes2bit(rle))

            num = rle_input.read(32)
            word_size = rle_input.read(5) + 1
            rle_sizes = [rle_input.read(4) + 1 for _ in range(4)]

            i = 0
            out = np.zeros(num, dtype=np.uint8)
            while i < num:
                x = rle_input.read(1)
                j = i + 1 + rle_input.read(rle_sizes[rle_input.read(2)])
                if x:
                    val = rle_input.read(word_size)
                    out[i:j] = val
                    i = j
                else:
                    while i < j:
                        val = rle_input.read(word_size)
                        out[i] = val
                        i += 1

            image = np.reshape(out, [height, width, 4])[:, :, 3]
            return image


        class_labels = {}
        class_label_names = []
        class_number = 0  

        for i in range(0, len(df), 1):
            if not pd.isna(df['Tag'][i]):
                all_coordinates = []

                json_data = json.loads(df['Tag'][i])
                original_width = json_data[0]['original_width']
                original_height = json_data[0]['original_height']

                # Extract the image name from the "image" column
                image_name = df['image'][i].split('/')[-1].split('.')[0]

                for j in range(0, len(json_data), 2):
                    label_data = json_data[j]
                    rle_data = json_data[j + 1]['rle']

                    class_label = label_data['keypointlabels'][0].strip("[]")

                    # Check if the class label has been encountered before
                    if class_label not in class_labels:
                        class_labels[class_label] = class_number
                        class_label_names.append(class_label)
                        class_number += 1

                    numerical_label = class_labels[class_label]

                    mask = rle_to_mask(rle_data, original_height, original_width)

                    H, W = mask.shape[0:2]

                    #gray_img = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                    ret, bin_img = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    contours, _ = cv2.findContours(bin_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)

                    
                    for j in contours:
                        class_coordinates = []
                        pre = j[0]
                        for i in j:
                            if abs(i[0][0] - pre[0][0]) > 1 or abs(i[0][1] - pre[0][1]) > 1:
                                pre = i
                                temp = list(i[0])
                                temp[0] /= W
                                temp[1] /= H
                                class_coordinates.extend(temp)
                    all_coordinates.append((numerical_label, class_coordinates))

                image_filename = f'{image_name}.txt'
                with open(os.path.join(labels_dir, image_filename), 'w') as txt_file:
                    for coordinates in all_coordinates:
                        numerical_label, coordinates_str = coordinates
                        txt_file.write(f"{numerical_label} {' '.join(map(str, coordinates_str))}\n")

                print(f"Coordinates for image {image_name} saved to '{image_filename}'")
        
        yaml_data = {
            'train': '../train/images',
            'test': '../test/images',
            'val': '../val/images',
            'nc': len(class_labels),
            'names': [f"'{label}'" for label in class_label_names]
        }


        with open(os.path.join(main_dir, 'data.yaml'), 'w') as yaml_file:
            yaml_file.write("names: [")
            yaml_file.write(", ".join(yaml_data['names']))
            yaml_file.write("]\n")
            del yaml_data['names']
            yaml.dump(yaml_data, yaml_file)

        source_directory = combined_path
        destination_directory = images_dir

        supported_extensions = (".png", ".jpg", ".jpeg", ".webp", ".svg")
        image_files = [f for f in os.listdir(source_directory) if f.lower().endswith(supported_extensions)]

        # Print the list of image files found
        print("Image files in the source directory:")
        for image_file in image_files:
            print(image_file)

        # Copy images
        for image_file in image_files:
            source_path = os.path.join(source_directory, image_file)
            destination_path = os.path.join(destination_directory, image_file)
            
            shutil.copy2(source_path, destination_path)

        print("Images copied successfully.")

        split_data(main_dir, images_dir, labels_dir)
        
        zip_filename = 'Yolo_Dataset.zip'

        shutil.make_archive(main_dir, 'zip', main_dir)
            
        with open(f"{main_dir}.zip", 'rb') as zip_file:
            response = HttpResponse(zip_file.read(), content_type="application/zip")
            response['Content-Disposition'] = f'attachment; filename={zip_filename}'
            return response
        
    else:
        return JsonResponse({'error': 'Invalid request'}, status=400)
    