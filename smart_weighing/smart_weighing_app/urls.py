from django.urls import path
from . import views

urlpatterns = [
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),
    path('home/', views.home, name='home'),  # Home Page
    path('load-cell/', views.load_cell_value, name='load_cell_value'),
    path('set_price/', views.set_price, name='set_price'),
    path('item_list/', views.item_list, name='item_list'),  # For managing items
    path('item_add/', views.add_item, name='item_add'),
    path('item_edit/<int:item_id>/', views.edit_item, name='item_edit'),
    path('item_delete/<int:item_id>/', views.delete_item, name='item_delete'),
    path('register/', views.register_shop, name='register_shop'),  # For creating a new shop
    path('register/<int:shop_id>/', views.register_shop, name='edit_shop'),  # For editing an existing shop
    path('shop_list/', views.shop_list, name='shop_list'),  # 👈 This is what Django needs
    path('delete/<int:shop_id>/', views.delete_shop, name='delete_shop'),
    path('price_list/', views.price_list, name='price_list'),
    path('price_add', views.price_add, name='price_add'),
    path('price_edit/<int:pk>/', views.price_edit, name='price_edit'),
    path('price_delete/<int:pk>/', views.price_delete, name='price_delete'),
    path('error/', views.error_page, name='error_page'),  # Add this line
    path('generate_bill/<int:bill_id>/', views.generate_bill, name='generate_bill'),  # Include bill_id in the URL
    path('bills/', views.bill_list, name='bill_list'),
]
