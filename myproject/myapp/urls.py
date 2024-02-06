# myapp/urls.py
from django.urls import path
from .views import naive_bayes, k_means_clustering, svm_function, Linear_Regression, Logistic_Regression,RandomForest
from django.views.static import serve
from django.conf import settings

urlpatterns = [
    path('static/<path:path>/', serve, {'document_root': settings.STATICFILES_DIRS[0]}),
    path('', serve, {'document_root': settings.STATICFILES_DIRS[0], 'path': 'index.html'}),
    path('naive_bayes/', naive_bayes, name='naive_bayes'),
    path('kmeans/', k_means_clustering, name='k_means_clustering'),
    path('svm/', svm_function, name='svm'),
    path('Linear_Regression/', Linear_Regression, name='Linear_Regression'),
    path('Logistic_Regression/', Logistic_Regression, name='Logistic_Regression'),
    path('RandomForest/', RandomForest, name='RandomForest'),
]


