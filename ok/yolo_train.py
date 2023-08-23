# train_yolov8.py

import os
from ultralytics import YOLO

def train_yolov8_on_custom_dataset(yaml_file):
    try:
        # Save the uploaded YAML file
        yaml_path = os.path.join('uploads', yaml_file.name)
        with open(yaml_path, 'wb') as f:
            for chunk in yaml_file.chunks():
                f.write(chunk)

        # Load the model
        model = YOLO('yolov8s.pt')

        # Training
        results = model.train(
            data=yaml_path,
            imgsz=240,
            epochs=10,
            batch=8,
            name='yolov8s_custom'
        )
        output_dir = os.path.join('model_results', 'yolov8n_custom')
        os.makedirs(output_dir, exist_ok=True)
        results.save(output_dir)

        return {'message': 'Training completed successfully', 'results': results}
    except Exception as e:
        return {'error': str(e)}
