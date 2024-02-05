# myapp/urls.py
# myapp/urls.py
from django.urls import path
from .views import  Linear_Regression
from django.views.static import serve
from django.conf import settings


urlpatterns = [
    path('static/<path:path>/', serve, {'document_root': settings.STATICFILES_DIRS[0]}),
    path('', serve, {'document_root': settings.STATICFILES_DIRS[0], 'path': 'index.html'}),
    path('Linear_Regression/', Linear_Regression, name='Linear_Regression')

]
