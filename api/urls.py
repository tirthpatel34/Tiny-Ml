from django.urls import path
from .views import predict_leaf

urlpatterns = [
    path("predict/", predict_leaf, name="predict_leaf"),
]
