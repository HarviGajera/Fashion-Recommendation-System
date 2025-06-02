from django.db import models
from django.contrib.auth.models import User


class Product(models.Model):
    product_name = models.CharField(max_length=300)
    category = models.CharField(max_length=255)
    image = models.ImageField(upload_to='products/')
    rating = models.FloatField(null=True, blank=True)
    image_embedding = models.BinaryField()
    text_embedding = models.BinaryField()

    def __str__(self):
        return self.product_name

class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    profile_picture = models.ImageField(upload_to='profile_pics/', blank=True, null=True)
    favorite_products = models.ManyToManyField(Product, blank=True)
    
    def __str__(self):
        return self.user.username
    
class Profile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    profile_picture = models.ImageField(upload_to='profile_pics/', null=True, blank=True)
    bio = models.TextField(blank=True)
    gender = models.CharField(max_length=20, blank=True)
    age = models.PositiveIntegerField(null=True, blank=True)
    
    def __str__(self):
        return f"{self.user.username}'s Profile"
    
class SearchHistory(models.Model):
    SEARCH_TYPE_CHOICES = [
        ('text', 'Text'),
        ('image', 'Image'),
    ]

    user = models.ForeignKey(User, on_delete=models.CASCADE)
    query = models.CharField(max_length=255, blank=True)  # Text query or image label
    image = models.ImageField(upload_to='history_images/', blank=True, null=True)  # Image preview if any
    search_type = models.CharField(
        max_length=5,
        choices=[('text', 'Text'), ('image', 'Image')],
        default='text'
    )
    timestamp = models.DateTimeField(auto_now_add=True)