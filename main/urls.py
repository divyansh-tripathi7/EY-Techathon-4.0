from django.urls import path, re_path
from django.urls.conf import include
from .views import home

urlpatterns= [
    path('', home, name='home'),
]
