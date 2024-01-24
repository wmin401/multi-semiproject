# myapp/urls.py
# myapp/urls.py
from django.urls import path
from .views import get_api_view, post_api_view
from django.views.static import serve
from django.conf import settings

urlpatterns = [
    path('get-api/', get_api_view, name='get-api'),
    path('post-api/', post_api_view, name='post-api'),
    path('static/<path:path>/', serve, {'document_root': settings.STATICFILES_DIRS[0]}),
    path('', serve, {'document_root': settings.STATICFILES_DIRS[0], 'path': 'index.html'}),
]
