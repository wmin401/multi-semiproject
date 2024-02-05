# myapp/urls.py
# myapp/urls.py
from django.urls import path
from .views import k_means_clustering
from django.views.static import serve
from django.conf import settings

urlpatterns = [
    path('static/<path:path>/', serve, {'document_root': settings.STATICFILES_DIRS[0]}),
    path('', serve, {'document_root': settings.STATICFILES_DIRS[0], 'path': 'index.html'}),
    path('kmeans/', k_means_clustering, name='k_means_clustering'),
]
