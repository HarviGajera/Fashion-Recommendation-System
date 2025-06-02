from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('', views.home, name='home'),
    path('search/image/', views.search_by_image, name='search_by_image'),
    path('search/text/', views.search_by_text, name='search_by_text'),
    path('trending-search/', views.trending_image_search, name='trending_image_search'),
    
    # Favourites
    path('toggle-favorite/', views.toggle_favorite, name='toggle_favorite'),
    
    # History
    path('clear-history/', views.clear_history, name='clear_history'),
    
    # Authentication URLs
    path('login/', views.user_login, name='login'),
    path('signup/', views.user_signup, name='signup'),
    path('logout/', views.user_logout, name='logout'),
    
    # User Profile URLs
    path('profile/', views.user_profile, name='profile'),
    path('edit-profile/', views.edit_profile, name='edit_profile'),
    path('add-to-favorites/<int:product_id>/', views.add_to_favorites, name='add_to_favorites'),
    
    # Settings
    path('settings/', views.settings_view, name='settings'),
    path('settings/change-username/', views.change_username, name='change_username'),
    path('change-password/', views.change_password, name='change_password'),
]

# Serve media files in development
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)