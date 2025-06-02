import os
import pandas as pd
import requests
from io import BytesIO
from PIL import Image
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel
from sentence_transformers import SentenceTransformer # type: ignore
from django.conf import settings
from tqdm import tqdm

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Batch size for processing
BATCH_SIZE = 16  # Smaller batch size because CLIP is more memory intensive

# Initialize models
clip_model_name = "openai/clip-vit-base-patch32"
clip_model = CLIPModel.from_pretrained(clip_model_name)
clip_processor = CLIPProcessor.from_pretrained(clip_model_name)

# Initialize Sentence Transformer model for text embeddings
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')  # 384 dimensions

# CNN Alignment model for mapping text embeddings to image embedding space
class AlignmentModel(nn.Module):
    def __init__(self, text_embedding_dim=384, image_embedding_dim=512, hidden_dim=512):
        super(AlignmentModel, self).__init__()
        self.fc1 = nn.Linear(text_embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, image_embedding_dim)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# Create and initialize the alignment model
alignment_model = AlignmentModel()

# Path to save the alignment model
ALIGNMENT_MODEL_PATH = os.path.join(settings.BASE_DIR, 'alignment_model.pth')

# Helper function to clean file names
def clean_filename(name):
    """Clean a filename by removing any non-alphanumeric characters"""
    return "".join(c if c.isalnum() else "_" for c in name)

# Download image and process it
def download_and_process_image(image_url, product_name, media_folder):
    """Download an image from a URL and save it to disk"""
    try:
        response = requests.get(image_url, timeout=10)

        img = Image.open(BytesIO(response.content))
        
        # Convert to RGB if needed (CLIP expects RGB images)
        if img.mode != 'RGB':
            img = img.convert('RGB')
            
        filename = clean_filename(product_name) + '.jpg'
        image_path = os.path.join(media_folder, filename)
        img.save(image_path)
        return image_path
    except Exception as e:
        return None

# Generate CLIP image embeddings in batches
def batch_generate_clip_image_embeddings(image_paths):
    """Generate image embeddings using CLIP model in batches"""
    embeddings = []
    
    # Create a tqdm progress bar
    for i in tqdm(range(0, len(image_paths), BATCH_SIZE), desc="Generating image embeddings", total=(len(image_paths) + BATCH_SIZE - 1) // BATCH_SIZE):
        batch_paths = image_paths[i:i+BATCH_SIZE]
        batch_images = []
        
        for path in batch_paths:
            try:
                img = Image.open(path)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                batch_images.append(img)
            except Exception as e:
                print(f"Error processing image {path}: {e}")
                # Use a blank image as fallback
                blank_img = Image.new('RGB', (224, 224), color='white')
                batch_images.append(blank_img)
        
        # Process images through CLIP
        with torch.no_grad():
            inputs = clip_processor(images=batch_images, return_tensors="pt", padding=True)
            outputs = clip_model.get_image_features(**inputs)
            
            # Normalize embeddings (important for CLIP)
            batch_embeddings = F.normalize(outputs, p=2, dim=1).cpu().numpy()
            
            for embedding in batch_embeddings:
                embeddings.append(embedding.flatten())
    
    return embeddings

# Generate single image embedding using CLIP
def generate_clip_image_embedding(image_input):
    """Generate image embedding for a single image using CLIP"""
    try:
        if isinstance(image_input, str):  # It's a path
            img = Image.open(image_input)
        else:
            img = Image.open(image_input)
            
        if img.mode != 'RGB':
            img = img.convert('RGB')
            
        # Process image through CLIP
        with torch.no_grad():
            inputs = clip_processor(images=[img], return_tensors="pt")
            outputs = clip_model.get_image_features(**inputs)
            
            # Normalize embedding
            embedding = F.normalize(outputs, p=2, dim=1).cpu().numpy().flatten()
        
        return embedding
    except Exception as e:
        print(f"Error generating image embedding: {e}")
        return np.zeros(512)  # Return zeros as fallback

# Helper function to prepare rich text description for embedding
def prepare_product_text(product_name, category="", color="", meta_keywords=""):
    """Combine product attributes into a rich text description for embedding"""
    text_parts = [product_name]
    
    # Add category if available
    if category and isinstance(category, str) and category.strip():
        text_parts.append(f"Category: {category.strip()}")
    
    # Add color if available
    if color and isinstance(color, str) and color.strip():
        text_parts.append(f"Color: {color.strip()}")
    
    # Add keywords if available
    if meta_keywords and isinstance(meta_keywords, str) and meta_keywords.strip():
        text_parts.append(f"Keywords: {meta_keywords.strip()}")
    
    # Join all parts with spaces
    return " ".join(text_parts)

# Generate SentenceTransformer text embeddings in batches
def batch_generate_text_embeddings(texts, categories=None, colors=None, meta_keywords=None):
    """Generate text embeddings using SentenceTransformer model in batches"""
    embeddings = []
    rich_texts = []
    
    # Prepare rich text descriptions by combining product attributes
    for i, text in enumerate(texts):
        category = categories[i] if categories is not None and i < len(categories) else ""
        color = colors[i] if colors is not None and i < len(colors) else ""
        keyword = meta_keywords[i] if meta_keywords is not None and i < len(meta_keywords) else ""
        
        rich_text = prepare_product_text(text, category, color, keyword)
        rich_texts.append(rich_text)
    
    # Create a tqdm progress bar
    for i in tqdm(range(0, len(rich_texts), BATCH_SIZE), desc="Generating text embeddings", total=(len(rich_texts) + BATCH_SIZE - 1) // BATCH_SIZE):
        batch_texts = rich_texts[i:i+BATCH_SIZE]
        
        try:
            # Generate embeddings directly with sentence_model
            batch_embeddings = sentence_model.encode(batch_texts)
            
            # Add each embedding to our list
            for embedding in batch_embeddings:
                embeddings.append(embedding)
                
        except Exception as e:
            print(f"Error generating text embeddings: {e}")
            # Use zeros as fallback
            for _ in range(len(batch_texts)):
                embeddings.append(np.zeros(384))  # all-MiniLM-L6-v2 has 384 dimensions
    
    return embeddings

# Generate a single text embedding 
def generate_text_embedding(text, category="", color="", meta_keywords=""):
    """Generate text embedding for a single text using SentenceTransformer"""
    try:
        # Combine all product attributes into rich text description
        rich_text = prepare_product_text(text, category, color, meta_keywords)
        embedding = sentence_model.encode([rich_text])[0]
        return embedding
    except Exception as e:
        print(f"Error generating text embedding: {e}")
        return np.zeros(384)  # Return zeros as fallback

# Align text embeddings to image space using the CNN model
def align_text_to_image_space(text_embeddings):
    """Transform text embeddings to align with image embedding space with improved error handling"""
    try:
        # Create the model with correct dimensions
        if not text_embeddings or len(text_embeddings) == 0:
            print("No text embeddings provided to align_text_to_image_space")
            return None
            
        # Get dimension from the first embedding
        text_dim = len(text_embeddings[0])
        print(f"Text embedding dimension: {text_dim}")
        
        # Create the model with the correct input dimension
        model = AlignmentModel(text_embedding_dim=text_dim)
        
        # Check if the model file exists and load it
        if os.path.exists(ALIGNMENT_MODEL_PATH):
            try:
                model.load_state_dict(torch.load(ALIGNMENT_MODEL_PATH))
                print(f"Loaded alignment model from {ALIGNMENT_MODEL_PATH}")
            except Exception as e:
                print(f"Error loading model, will create a new one: {e}")
                # Save the newly created model
                torch.save(model.state_dict(), ALIGNMENT_MODEL_PATH)
                print(f"Saved new alignment model to {ALIGNMENT_MODEL_PATH}")
        else:
            # If model doesn't exist, create a new one and save it
            print(f"Alignment model file not found at {ALIGNMENT_MODEL_PATH}, creating a new one")
            torch.save(model.state_dict(), ALIGNMENT_MODEL_PATH)
        
        # Set model to evaluation mode
        model.eval()
        
        aligned_embeddings = []
        
        # Process in small batches to avoid memory issues
        BATCH_SIZE = 16
        for i in range(0, len(text_embeddings), BATCH_SIZE):
            batch_embeddings = text_embeddings[i:i+BATCH_SIZE]
            
            # Convert list of embeddings to tensor
            input_tensor = torch.tensor(np.vstack(batch_embeddings), dtype=torch.float32)
            
            with torch.no_grad():
                # Forward pass
                outputs = model(input_tensor)
                
                # Normalize outputs for cosine similarity
                outputs = F.normalize(outputs, p=2, dim=1)
                
                # Convert to numpy and add to results
                batch_aligned = outputs.cpu().numpy()
                
                for aligned in batch_aligned:
                    aligned_embeddings.append(aligned)
        
        return aligned_embeddings
    
    except Exception as e:
        print(f"Error in align_text_to_image_space: {e}")
        import traceback
        traceback.print_exc()
        return None

# Train the alignment model on a batch of data
def train_alignment_model(text_embeddings, image_embeddings, model=alignment_model, 
                          epochs=10, batch_size=32, lr=0.001):
    """Train the alignment model to map text embeddings to image embedding space"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CosineEmbeddingLoss()
    
    # Convert to numpy arrays for consistent handling
    text_embeddings = np.vstack(text_embeddings)
    image_embeddings = np.vstack(image_embeddings)
    
    # Convert to tensors
    text_tensor = torch.tensor(text_embeddings, dtype=torch.float32)
    image_tensor = torch.tensor(image_embeddings, dtype=torch.float32)
    
    # Create dataset
    dataset = torch.utils.data.TensorDataset(text_tensor, image_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for text_batch, image_batch in dataloader:
            text_batch, image_batch = text_batch, image_batch
            
            # Forward pass
            aligned_text = model(text_batch)
            
            # Compute loss
            target = torch.ones(text_batch.size(0))  # Target is 1 for cosine similarity
            loss = criterion(aligned_text, image_batch, target)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")
    
    # Save the trained model
    torch.save(model.state_dict(), ALIGNMENT_MODEL_PATH)
    
    return model

# Process CSV and store data with batch processing
def process_csv_and_store_data(csv_path, media_folder):
    """Process CSV data, download images, generate embeddings, and prepare for DB storage"""
    print(f"Processing CSV file: {csv_path}")
    df = pd.read_csv(csv_path)
    
    valid_rows = []
    image_paths = []
    product_names = []
    categories = []
    colors = []
    meta_keywords = []
    
    # First, download and process all images
    print("Downloading and processing images...")
    for index, row in tqdm(df.iterrows(), total=len(df)):
        image_url = row['medium']
        product_name = row['product_name']
        category = row.get('grouped_category', '')
        color = row.get('color', '')
        meta_keyword = row.get('meta_keywords', '')
        
        image_path = download_and_process_image(image_url, product_name, media_folder)
        if image_path:
            image_paths.append(image_path)
            product_names.append(product_name)
            categories.append(category)
            colors.append(color)
            meta_keywords.append(meta_keyword)
    
    # Then, generate embeddings in batches
    print(f"Generating CLIP image embeddings for {len(image_paths)} products...")
    image_embeddings = batch_generate_clip_image_embeddings(image_paths)
    
    print(f"Generating text embeddings for {len(product_names)} products...")
    text_embeddings = batch_generate_text_embeddings(product_names, categories, colors, meta_keywords)
    
    # Train the alignment model
    print("Training alignment model...")
    train_alignment_model(text_embeddings, image_embeddings)
    
    # Generate aligned text embeddings
    print("Aligning text embeddings to image space...")
    aligned_text_embeddings = align_text_to_image_space(text_embeddings)
    
    # Combine all data
    print("Combining data...")
    for i in range(len(image_paths)):
        valid_rows.append({
            'product_name': product_names[i],
            'category': categories[i],
            'color': colors[i],
            'meta_keywords': meta_keywords[i],
            'image': image_paths[i],
            'image_embedding': image_embeddings[i],
            'text_embedding': aligned_text_embeddings[i]  # Save aligned text embeddings
        })
    
    # Create a new CSV with valid rows
    print(f"Saving data for {len(valid_rows)} products...")
    valid_df = pd.DataFrame(valid_rows)
    valid_df.to_csv('valid_products.csv', index=False)
    
    return valid_rows

# Search for similar products (for single query)
def find_similar_products(query_embedding, index, k=5):
    """Find k most similar products using FAISS index"""
    # Ensure the embedding is the right shape and type
    if isinstance(query_embedding, list):
        query_embedding = np.array(query_embedding)
    
    query_embedding = query_embedding.reshape(1, -1).astype('float32')
    
    # Search the index
    D, I = index.search(query_embedding, k)
    
    return I[0]  # Return indices of similar items

import pandas as pd

def load_rating_map(csv_path):
    try:
        df = pd.read_csv(csv_path)
        df['product_name'] = df['product_name'].astype(str).str.strip()  # Clean whitespace
        rating_map = dict(zip(df['product_name'], df['rating']))
        return rating_map
    except Exception as e:
        print(f"Error loading rating map from {csv_path}: {e}")
        return {}