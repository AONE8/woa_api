from django.urls import path

from .views import WoaSelectionView

# визначення схеми кінцевих точок для api/
urlpatterns = [
    path("woa-selection/", WoaSelectionView.as_view())
]
