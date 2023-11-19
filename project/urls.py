from django.contrib import admin
from django.urls import path, include, re_path
from django.conf import settings
from django.conf.urls.static import static
from ok.views import yolo, home, yolo_api,anomalib_config,yolo_csv, webcam, model_upload_anomalib, model_upload_yolo, anomaly_detection_api,anomalib_webcam, process_directory, train_yolo, train_anomaly_detection, yolo_webcam
urlpatterns = [
    path('admin/', admin.site.urls),
    path('', home, name='home'),
    path('yolo/', yolo, name='yolo'),
    #path('anomaly-detection/', anomaly_detection, name='anomaly_detection'),
    #path('detect-anomaly/', anomaly_detection, name='detect_anomaly'),
    path('api/yolo/', yolo_api, name='yolo_api'),
    path('api/anomaly_detection/', anomaly_detection_api, name='anomaly_detection_api'),
    path('process_directory/', process_directory, name='process_directory'),
    path('train/', train_yolo, name='train_yolo'),
    path('train_anomaly_detection/', train_anomaly_detection, name='train_anomaly_detection'),
    path('yolo_webcam/',yolo_webcam, name='yolo_webcam'),
    path('anomalib_webcam/',anomalib_webcam, name='anomalib_webcam'),
    path('model_upload_yolo/',model_upload_yolo, name='model_upload_yolo'),
    path('model_upload_anomalib/',model_upload_anomalib, name='model_upload_anomalib'),
    path('webcam/', webcam, name='webcam'),
    path('anomalib_config/', anomalib_config, name='anomalib_config'),
    path('yolo_csv/', yolo_csv, name='yolo_csv'),



    # include other app URLs if needed
] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
