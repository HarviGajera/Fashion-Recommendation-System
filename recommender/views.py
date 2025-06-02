from django.shortcuts import render, redirect, get_object_or_404
from .faiss_index import load_faiss_indices, search_similar_items
from .models import Product, UserProfile, Profile, SearchHistory
from django.http import JsonResponse
from .utils import generate_clip_image_embedding, generate_text_embedding, align_text_to_image_space
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login, logout, update_session_auth_hash
from django.contrib import messages
from django.contrib.auth.decorators import login_required
import numpy as np
import os
from django.conf import settings
import uuid
from .utils import load_rating_map

# Home view
def home(request):
    return render(request, 'home.html')


csv_path = os.path.join(settings.BASE_DIR, 'fashion_data.csv')
rating_map = load_rating_map(csv_path)


def search_by_image(request):
    if not request.user.is_authenticated:
        messages.info(request, "Please Login/Signup First")
        return redirect('login')
    
    if request.method == "POST" and request.FILES.get('image'):
        try:
            uploaded_image = request.FILES['image']
            
            # Generate image embedding using CLIP
            image_embedding = generate_clip_image_embedding(uploaded_image)
            
            # Load FAISS indices
            image_index, _, product_ids = load_faiss_indices()
            
            if image_index is None:
                return render(request, 'results.html', {'error_message': 'Error loading search index. Please try again later.'})
            
            # Search for similar products with deduplication
            indices = search_similar_items(image_embedding, image_index, k=5, max_results=50)
            
            if indices is None:
                return render(request, 'results.html', {'error_message': 'No similar products found.'})
            
            # Get products from database
            similar_products = Product.objects.filter(id__in=indices.tolist())
            
            # Final check for uniqueness in Python code (in case database has duplicates)
            unique_products = []
            unique_product_names = set()
            
            for product in similar_products:
                # Check if this product name is already included
                if product.product_name not in unique_product_names:
                    unique_products.append(product)
                    unique_product_names.add(product.product_name)
                    
                    # Stop once we have 5 unique products
                    if len(unique_products) >= 5:
                        break
            
            if not unique_products:
                return render(request, 'results.html', {'error_message': 'No similar products found.'})
            
            # Get Ratings fro CSV file
            for product in unique_products:
                try:
                    product_name = product.product_name.strip()
                    rating = rating_map.get(product_name)
                    print(f"[DEBUG] Matching '{product_name}' -> Rating: {rating}")
                    product.rating = rating if rating is not None else 0
                except Exception as e:
                    print(f"[ERROR] While assigning rating to '{product.product_name}': {e}")
                    product.rating = 0
            
            # Check if user is authenticated and get their favorites
            favorites = []
            if request.user.is_authenticated:
                user_profile = UserProfile.objects.get(user=request.user)
                favorites = [product.id for product in user_profile.favorite_products.all()]
            
            # Store the uploaded image temporarily for display
            if hasattr(uploaded_image, 'temporary_file_path'):
                # For file uploads handled by Django
                request.session['search_image_url'] = request.build_absolute_uri(uploaded_image.url)
            else:
                # Save the image to a temporary location for display
                # Create a unique filename
                filename = f"search_image_{uuid.uuid4().hex}.{uploaded_image.name.split('.')[-1]}"
                temp_dir = os.path.join(settings.MEDIA_ROOT, 'temp_searches')
                
                # Create directory if it doesn't exist
                if not os.path.exists(temp_dir):
                    os.makedirs(temp_dir)
                    
                # Save the file
                temp_path = os.path.join(temp_dir, filename)
                with open(temp_path, 'wb+') as destination:
                    for chunk in uploaded_image.chunks():
                        destination.write(chunk)
                
                # Store URL in session
                request.session['search_image_url'] = f"{settings.MEDIA_URL}temp_searches/{filename}"
                
                # For Search History
                SearchHistory.objects.create(
                    user=request.user,
                    query='',
                    search_type='image',
                    image=f"temp_searches/{filename}"  # Store relative path to the image
                )
                                
            return render(request, 'results.html', {
                'products': unique_products,
                'favorites': favorites
            })
            
        except Exception as e:
            return render(request, 'results.html', {'error_message': f"Error: {str(e)}"})

def search_by_text(request):
    if not request.user.is_authenticated:
        messages.info(request, "Please Login/Signup First")
        return redirect('login')
    
    if request.method == "POST" and request.POST.get('text'):
        try:
            text_input = request.POST.get('text')
            
            category = request.POST.get('category', '')
            color = request.POST.get('color', '')
            keywords = request.POST.get('keywords', '')
            
            # Log the search parameters
            print(f"Text search: '{text_input}', Category: '{category}', Color: '{color}', Keywords: '{keywords}'")
            
            # Generate text embedding
            text_embedding = generate_text_embedding(text_input, category, color, keywords)
            
            # Debug step - ensure text embedding is not None and is properly shaped
            if text_embedding is None or not isinstance(text_embedding, np.ndarray):
                return render(request, 'results.html', {'error_message': 'Failed to generate text embedding'})
            
            # Load FAISS indices directly
            _, text_index, product_ids = load_faiss_indices()
            
            if text_index is None:
                return render(request, 'results.html', {'error_message': 'Error loading search index. Please try again later.'})
            
            # Get index dimension
            index_dim = text_index.d
            embedding_dim = text_embedding.shape[0]
            
            print(f"Text embedding dimension: {embedding_dim}, Index dimension: {index_dim}")
            
            # If dimensions don't match and alignment is needed
            if embedding_dim != index_dim:
                if embedding_dim == 384 and index_dim == 512:  # Need alignment
                    print("Aligning text embedding to match index dimension")
                    # Align text embedding to image space
                    aligned_embeddings = align_text_to_image_space([text_embedding])
                    
                    if aligned_embeddings is None or len(aligned_embeddings) == 0:
                        return render(request, 'results.html', {'error_message': 'Failed to align text embedding'})
                    
                    # Get the first (and only) embedding from the list
                    text_embedding = aligned_embeddings[0]
                else:
                    # Dimensions don't match and can't be aligned
                    return render(request, 'results.html', 
                                {'error_message': f'Embedding dimension mismatch: {embedding_dim} vs {index_dim}'})
            
            # Ensure embedding is normalized and float32
            norm = np.linalg.norm(text_embedding)
            if norm > 0:
                text_embedding = text_embedding / norm
            text_embedding = text_embedding.astype(np.float32)
            
            # Search for similar products with deduplication - request more results to handle duplicates
            indices = search_similar_items(text_embedding, text_index, k=5, max_results=100)
            
            if indices is None or len(indices) == 0:
                return render(request, 'results.html', {'error_message': 'No similar products found.'})
            
            # Get products from database
            similar_products = Product.objects.filter(id__in=indices.tolist())
            
            # Get Ratings fro CSV file
            unique_products = []
            unique_product_names = set()
            
            for product in similar_products:
                # Check if this product name is already included
                if product.product_name not in unique_product_names:
                    unique_products.append(product)
                    unique_product_names.add(product.product_name)
                    
                    # Stop once we have 5 unique products
                    if len(unique_products) >= 5:
                        break
            
            if not unique_products:
                return render(request, 'results.html', {'error_message': 'No similar products found in database.'})
            
            print(f"Found {len(unique_products)} unique similar products")
            
            # Attach rating to each product dynamically
            for product in unique_products:
                try:
                    product_name = product.product_name.strip()
                    rating = rating_map.get(product_name)
                    print(f"[DEBUG] Matching '{product_name}' -> Rating: {rating}")
                    product.rating = rating if rating is not None else 0
                except Exception as e:
                    print(f"[ERROR] While assigning rating to '{product.product_name}': {e}")
                    product.rating = 0
            
            # Check if user is authenticated and get their favorites
            favorites = []
            if request.user.is_authenticated:
                user_profile = UserProfile.objects.get(user=request.user)
                favorites = [product.id for product in user_profile.favorite_products.all()]
                
            # Clear any previous search image URL from session
            if 'search_image_url' in request.session:
                del request.session['search_image_url']
                
            # For Search History
            SearchHistory.objects.create(user=request.user, query=text_input, search_type='text')
                
            return render(request, 'results.html', {
                'products': unique_products,
                'favorites': favorites,
                'search_term': text_input
            })
            
        except Exception as e:
            # More detailed error message
            import traceback
            error_details = traceback.format_exc()
            return render(request, 'results.html', {'error_message': f"Error: {str(e)}\n{error_details}"})
        

    return render(request, 'results.html', {'error_message': 'Invalid search request'})

# API endpoint for image search with deduplication
def api_search_by_image(request):
    if request.method == "POST" and request.FILES.get('image'):
        try:
            uploaded_image = request.FILES['image']
            
            # Generate image embedding using CLIP
            image_embedding = generate_clip_image_embedding(uploaded_image)
            
            # Load FAISS indices
            image_index, _, product_ids = load_faiss_indices()
            
            if image_index is None:
                return JsonResponse({'error': 'Error loading search index'}, status=500)
            
            # Search for similar products
            indices = search_similar_items(image_embedding, image_index, k=5, max_results=50)
            
            if indices is None:
                return JsonResponse({'error': 'No similar products found'}, status=404)
            
            # Get products from database
            similar_products = Product.objects.filter(id__in=indices.tolist())
            
            # Ensure unique products by product name
            unique_products = []
            unique_product_names = set()
            
            for product in similar_products:
                if product.product_name not in unique_product_names:
                    unique_products.append(product)
                    unique_product_names.add(product.product_name)
                    
                    if len(unique_products) >= 5:
                        break
            
            if not unique_products:
                return JsonResponse({'error': 'No similar products found'}, status=404)
                
            # Get user favorites if authenticated
            favorites = []
            if request.user.is_authenticated:
                user_profile = UserProfile.objects.get(user=request.user)
                favorites = [product.id for product in user_profile.favorite_products.all()]
                
            # Format response
            products_data = []
            for product in unique_products:
                products_data.append({
                    'id': product.id,
                    'name': product.product_name,
                    'category': product.category,
                    'image_url': request.build_absolute_uri(product.image.url),
                    'is_favorite': product.id in favorites
                })
                
            return JsonResponse({'products': products_data})          
              
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    
    return JsonResponse({'error': 'Invalid request'}, status=400)

# API endpoint for text search with deduplication
def api_search_by_text(request):
    if request.method == "POST":
        try:
            text_input = request.POST.get('text', '')
            category = request.POST.get('category', '')
            color = request.POST.get('color', '')
            keywords = request.POST.get('keywords', '')
            
            if not text_input:
                return JsonResponse({'error': 'Text input is required'}, status=400)
            
            # Generate text embedding with enhanced method
            text_embedding = generate_text_embedding(text_input, category, color, keywords)
            
            # Align text embedding to image space
            aligned_embedding = align_text_to_image_space([text_embedding])[0]
            
            # Load FAISS indices
            _, text_index, product_ids = load_faiss_indices()
            
            if text_index is None:
                return JsonResponse({'error': 'Error loading search index'}, status=500)
            
            # Search for similar products with deduplication
            indices = search_similar_items(aligned_embedding, text_index, k=5, max_results=50)
            
            if indices is None or len(indices) == 0:
                return JsonResponse({'error': 'No similar products found'}, status=404)
            
            # Get products from database
            similar_products = Product.objects.filter(id__in=indices.tolist())
            
            # Ensure unique products by product name
            unique_products = []
            unique_product_names = set()
            
            for product in similar_products:
                if product.product_name not in unique_product_names:
                    unique_products.append(product)
                    unique_product_names.add(product.product_name)
                    
                    if len(unique_products) >= 5:
                        break
            
            if not unique_products:
                return JsonResponse({'error': 'No similar products found'}, status=404)
            
            # Get user favorites if authenticated
            favorites = []
            if request.user.is_authenticated:
                user_profile = UserProfile.objects.get(user=request.user)
                favorites = [product.id for product in user_profile.favorite_products.all()]
                
            # Format response
            products_data = []
            for product in unique_products:
                products_data.append({
                    'id': product.id,
                    'name': product.product_name,
                    'category': product.category,
                    'image_url': request.build_absolute_uri(product.image.url),
                    'is_favorite': product.id in favorites
                })
                
            return JsonResponse({'products': products_data})
            
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    
    return JsonResponse({'error': 'Invalid request'}, status=400)

# Login view
def user_login(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        user = authenticate(request, username=username, password=password)
        
        if user is not None:
            login(request, user)
            messages.success(request, f"Welcome back, {username}!")
            return redirect('home')
        else:
            return render(request, 'login.html', {'error_message': 'Invalid username or password.'})
    return render(request, 'login.html')

# Signup view
def user_signup(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        email = request.POST.get('email')
        password1 = request.POST.get('password1')
        password2 = request.POST.get('password2')
        
        # Check if passwords match
        if password1 != password2:
            return render(request, 'signup.html', {'error_message': 'Passwords do not match.'})
        
        # Check if username exists
        if User.objects.filter(username=username).exists():
            return render(request, 'signup.html', {'error_message': 'Username already exists.'})
        
        # Check if email exists
        if User.objects.filter(email=email).exists():
            return render(request, 'signup.html', {'error_message': 'This email is already registered. Please login.'})
        
        # Create user
        user = User.objects.create_user(username=username, email=email, password=password1)
        # Create user profile
        UserProfile.objects.create(user=user)
        Profile.objects.create(user=user)  # Create Profile model as well
        
        # Auto login after signup
        login(request, user)
        messages.success(request, f"Welcome to Fashion Recommender, {username}!")
        return redirect('home')
    
    return render(request, 'signup.html')

# Logout view
def user_logout(request):
    logout(request)
    return redirect('home')

@login_required
def user_profile(request):
    # Get both profile models - force refresh from database
    user_profile, created_up = UserProfile.objects.get_or_create(user=request.user)
    profile, created_p = Profile.objects.get_or_create(user=request.user)
    
    # Get fresh user data from database to ensure we have latest values
    user = User.objects.get(id=request.user.id)
    
    # Get favorite products
    favorites = user_profile.favorite_products.all()
    
    # Get Ratings fro CSV file
    for product in favorites:
                try:
                    product_name = product.product_name.strip()
                    rating = rating_map.get(product_name)
                    print(f"[DEBUG] Matching '{product_name}' -> Rating: {rating}")
                    product.rating = rating if rating is not None else 0
                except Exception as e:
                    print(f"[ERROR] While assigning rating to '{product.product_name}': {e}")
                    product.rating = 0
                    
    # Get Search History
    text_history = SearchHistory.objects.filter(user=request.user, search_type='text').order_by('-timestamp')[:20]
    image_history = SearchHistory.objects.filter(user=request.user, search_type='image').order_by('-timestamp')[:20]

    
    context = {
        'profile': profile, 
        'user_profile': user_profile, 
        'favorites': favorites,
        'user': user,  # Pass fresh user data to template
        'image_history' : image_history, 
        'text_history' : text_history,
    }
    
    return render(request, 'profile.html', context)

# Add to favorites - AJAX endpoint
@login_required
def toggle_favorite(request):
    if request.method == 'POST' and request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        product_id = request.POST.get('product_id')
        
        try:
            product = Product.objects.get(id=product_id)
            profile = UserProfile.objects.get(user=request.user)
            
            # Check if the product is already in favorites
            if product in profile.favorite_products.all():
                # Remove from favorites
                profile.favorite_products.remove(product)
                return JsonResponse({'status': 'success', 'action': 'removed'})
            else:
                # Add to favorites
                profile.favorite_products.add(product)
                return JsonResponse({'status': 'success', 'action': 'added'})
                
        except Product.DoesNotExist:
            return JsonResponse({'status': 'error', 'message': 'Product not found'}, status=404)
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)}, status=500)
    
    return JsonResponse({'status': 'error', 'message': 'Invalid request'}, status=400)

# Add to favorites - Legacy URL method
@login_required
def add_to_favorites(request, product_id):
    try:
        product = Product.objects.get(id=product_id)
        profile = UserProfile.objects.get(user=request.user)
        
        # Check if product is already in favorites
        if product in profile.favorite_products.all():
            # Remove from favorites
            profile.favorite_products.remove(product)
            messages.success(request, f"{product.product_name} removed from favorites.")
        else:
            # Add to favorites
            profile.favorite_products.add(product)
            messages.success(request, f"{product.product_name} added to favorites!")
            
        # Redirect back to results or referrer page
        referer = request.META.get('HTTP_REFERER')
        if referer:
            return redirect(referer)
        return redirect('results')
    except Exception as e:
        return render(request, 'results.html', {'error_message': f"Error: {str(e)}"})

# Remove from favorites
@login_required
def remove_from_favorites(request, product_id):
    try:
        product = Product.objects.get(id=product_id)
        profile = UserProfile.objects.get(user=request.user)
        profile.favorite_products.remove(product)
        messages.success(request, f"{product.product_name} removed from favorites.")
        return redirect('profile')
    except Exception as e:
        messages.error(request, f"Error: {str(e)}")
        return redirect('profile')

@login_required
def edit_profile(request):
    if request.method == 'POST':
        # Get or create both profile models
        user_profile, created_up = UserProfile.objects.get_or_create(user=request.user)
        profile, created_p = Profile.objects.get_or_create(user=request.user)
        
        # Update user information
        request.user.first_name = request.POST.get('first_name', '')
        request.user.last_name = request.POST.get('last_name', '')
        request.user.save()
        
        # Try to update age if it exists on User model
        age = request.POST.get('age', '')
        if age:
            try:
                profile.age = int(age)
            except ValueError:
                pass
        
        # Update profile information
        profile.bio = request.POST.get('bio', '')
        profile.gender = request.POST.get('gender', '')
        
        # Handle profile picture upload
        if 'profile_picture' in request.FILES:
            profile.profile_picture = request.FILES['profile_picture']
            # Also update the UserProfile picture to keep them in sync
            user_profile.profile_picture = request.FILES['profile_picture']
        
        profile.save()
        user_profile.save()
        
        messages.success(request, "Profile updated successfully!")
        return redirect('profile')
    
    return redirect('profile')

# Settings view
@login_required
def settings_view(request):
    return render(request, 'settings.html')

# Change username view
@login_required
def change_username(request):
    if request.method == 'POST':
        new_username = request.POST.get('new_username')
        password = request.POST.get('password')
        
        # Verify password
        user = authenticate(username=request.user.username, password=password)
        
        if not user:
            messages.error(request, "Invalid password. Username change failed.")
            return redirect('settings')
        
        # Check if new username already exists
        if User.objects.filter(username=new_username).exists():
            messages.error(request, "Username already exists. Please choose a different one.")
            return redirect('settings')
        
        # Update username
        user = request.user
        user.username = new_username
        user.save()
        
        messages.success(request, "Username changed successfully!")
        return redirect('settings')
    
    # If GET request, redirect to settings page
    return redirect('settings')

# Change password view
@login_required
def change_password(request):
    if request.method == 'POST':
        current_password = request.POST.get('current_password')
        new_password1 = request.POST.get('new_password1')
        new_password2 = request.POST.get('new_password2')
        
        # Check if passwords match
        if new_password1 != new_password2:
            messages.error(request, "New passwords do not match.")
            return redirect('settings')
        
        # Verify current password
        user = authenticate(username=request.user.username, password=current_password)
        if not user:
            messages.error(request, "Current password is incorrect.")
            return redirect('settings')
        
        # Change password
        user = request.user
        user.set_password(new_password1)
        user.save()
        
        # Update the session to prevent the user from being logged out
        update_session_auth_hash(request, user)
        
        messages.success(request, "Password changed successfully!")
        return redirect('settings')
    
    # If GET request, redirect to settings page
    return redirect('settings')

from django.views.decorators.csrf import csrf_exempt
@csrf_exempt
def trending_image_search(request):
    if not request.user.is_authenticated:
        messages.info(request, "Please Login/Signup First")
        return redirect('login')
    
    if request.method == "POST":
        image_name = request.POST.get("image_name", "")
        image_path = os.path.join(settings.TRENDING_ROOT, image_name)
        print(image_name)
        print(image_path)
        
        if not os.path.exists(image_path):
            return render(request, 'results.html', {'error_message': 'Trending image not found.'})
        
        try:
            with open(image_path, 'rb') as f:
                from django.core.files.uploadedfile import InMemoryUploadedFile
                import io
                img_bytes = io.BytesIO(f.read())
                uploaded_file = InMemoryUploadedFile(img_bytes, None, image_name, 'image/jpeg', img_bytes.getbuffer().nbytes, None)
                
                # Fake a request.FILES dict to reuse the search_by_image logic
                request.FILES['image'] = uploaded_file
                request.session['search_image_url'] = f"{settings.STATIC_URL}trending/{image_name}"

                return search_by_image(request)
        except Exception as e:
            return render(request, 'results.html', {'error_message': f"Error loading trending image: {str(e)}"})

    return redirect('home')

@login_required
def clear_history(request):
    SearchHistory.objects.filter(user=request.user).delete()
    return redirect('profile')