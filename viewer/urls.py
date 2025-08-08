from django.urls import path
from . import views

urlpatterns = [
    path('', views.DashboardView.as_view(), name='dashboard'),  # Make dashboard the homepage
    path('history/', views.history_view, name='history'),
    path("live/", views.live_stream_view, name="live_stream"),
    path("video_feed/", views.video_feed, name="video_feed"),
    path('live-feeds/', views.latest_feeds, name='latest_feeds'),
]

