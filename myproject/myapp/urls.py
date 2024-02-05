# myapp/urls.py
# myapp/urls.py
from django.urls import path
from django.views.static import serve
from django.conf import settings
from sklearn import svm
from .views import svm_function

urlpatterns = [
    path('static/<path:path>/', serve, {'document_root': settings.STATICFILES_DIRS[0]}),
    path('', serve, {'document_root': settings.STATICFILES_DIRS[0], 'path': 'index.html'}),
    path('svm/', svm_function, name='svm'),
]
