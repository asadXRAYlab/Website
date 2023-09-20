from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from ok.views import yolo, home, yolo_api, anomaly_detection_api, process_directory, train_yolo, train_anomaly_detection
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

    # include other app URLs if needed
] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
