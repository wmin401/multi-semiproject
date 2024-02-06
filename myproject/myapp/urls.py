# myapp/urls.py
# myapp/urls.py
from django.urls import path
from .views import naive_bayes, RandomForest
from .views import naive_bayes, RandomForest
from django.views.static import serve
from django.conf import settings

urlpatterns = [
    path('static/<path:path>/', serve, {'document_root': settings.STATICFILES_DIRS[0]}),
    path('', serve, {'document_root': settings.STATICFILES_DIRS[0], 'path': 'index.html'}),
    path('naive_bayes/', naive_bayes, name='naive_bayes'),
    path('RandomForest/', RandomForest, name='RandomForest'),
    ]
