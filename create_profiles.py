import os
import django

# Set up Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'fashion_recommendation.settings')
django.setup()

# Import models after setting up Django
from django.contrib.auth.models import User
from recommender.models import UserProfile

def create_missing_profiles():
    # Get all users
    users = User.objects.all()
    profiles_created = 0
    
    for user in users:
        # Try to get profile, create if it doesn't exist
        profile, created = UserProfile.objects.get_or_create(user=user)
        
        if created:
            profiles_created += 1
            print(f"Created profile for user: {user.username}")
    
    print(f"Total profiles created: {profiles_created}")
    print(f"Total users checked: {len(users)}")

if __name__ == "__main__":
    create_missing_profiles()