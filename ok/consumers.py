
import cv2
import asyncio
from channels.generic.websocket import AsyncWebsocketConsumer
from django.http import JsonResponse
from channels.layers import get_channel_layer
from django.views.generic.base import View

class LiveWebcamConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.accept()

        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()

            await self.send(frame_bytes)

            await asyncio.sleep(0.1)

    async def receive(self, text_data):
        if text_data == 'capture_frame':
            await self.capture_frame()

    async def capture_frame(self):
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        if ret:
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            await self.send(frame_bytes)

class CaptureFrameView(View):
    def post(self, request, *args, **kwargs):
        channel_layer = get_channel_layer()
        async_to_sync(channel_layer.send)("websocket.connect", {"type": "websocket.receive", "text": "capture_frame"})
        return JsonResponse({"message": "Frame captured and sent to WebSocket"})