from django.urls import path
from . import views

urlpatterns = [
    path("", views.history_view, name="history"), # Root view shows history
    path("live/", views.live_stream_view, name="live_stream"),
    path("video_feed/", views.video_feed, name="video_feed"),
]

