from django.core.management.base import BaseCommand
from django.conf import settings
import os
import numpy as np
from tqdm import tqdm
import traceback
import logging
from recommender.models import Product
from recommender.utils import generate_text_embedding
from recommender.faiss_index import build_faiss_index

# Set up logging
logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = 'Regenerate text embeddings for all products and rebuild FAISS indices'

    def add_arguments(self, parser):
        parser.add_argument('--batch-size', type=int, help='Batch size for DB operations', default=50)
        parser.add_argument('--limit', type=int, help='Limit number of products to process', default=None)
        parser.add_argument('--force-rebuild', action='store_true', help='Force rebuild of FAISS indices even if no embeddings changed')

    def handle(self, *args, **kwargs):
        batch_size = kwargs.get('batch_size')
        limit = kwargs.get('limit')
        force_rebuild = kwargs.get('force_rebuild')
        
        self.stdout.write(self.style.SUCCESS("Starting text embedding regeneration process..."))
        
        # Get all products from the database
        products = Product.objects.all()
        
        if limit:
            products = products[:limit]
            self.stdout.write(f"Limited to {limit} products")
        
        total_products = products.count()
        self.stdout.write(self.style.SUCCESS(f"Found {total_products} products to update"))
        
        # Process products in batches
        num_batches = (total_products + batch_size - 1) // batch_size
        updated_count = 0
        failed_count = 0
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, total_products)
            batch_products = products[start_idx:end_idx]
            
            self.stdout.write(f"Processing batch {batch_idx + 1}/{num_batches} ({start_idx} to {end_idx})")
            
            products_to_update = []
            
            for product in tqdm(batch_products, desc=f"Generating text embeddings for batch {batch_idx + 1}"):
                try:
                    # Generate new text embedding
                    new_text_embedding = generate_text_embedding(
                        product.product_name,
                        category=product.category,
                    )
                    
                    # Skip if text embedding generation failed
                    if new_text_embedding is None or not isinstance(new_text_embedding, np.ndarray):
                        self.stderr.write(self.style.ERROR(f"Failed to generate text embedding for product {product.id}"))
                        failed_count += 1
                        continue
                    
                    # Convert embedding to binary format for storage
                    text_binary = np.array(new_text_embedding, dtype=np.float32).tobytes()
                    
                    # Update the product's text embedding
                    product.text_embedding = text_binary
                    products_to_update.append(product)
                    updated_count += 1
                    
                except Exception as e:
                    self.stderr.write(self.style.ERROR(f"Error processing product {product.id}: {str(e)}"))
                    traceback.print_exc()
                    failed_count += 1
            
            # Bulk update the products
            if products_to_update:
                try:
                    Product.objects.bulk_update(products_to_update, ['text_embedding'])
                    self.stdout.write(f"Updated {len(products_to_update)} products in batch {batch_idx + 1}/{num_batches}")
                except Exception as e:
                    self.stderr.write(self.style.ERROR(f"Error performing bulk update: {str(e)}"))
                    traceback.print_exc()
        
        self.stdout.write(self.style.SUCCESS(f"Updated text embeddings for {updated_count} products. Failed: {failed_count}"))
        
        # Rebuild FAISS indices if embeddings were updated or if force_rebuild is set
        if updated_count > 0 or force_rebuild:
            self.stdout.write(self.style.SUCCESS("Rebuilding FAISS indices..."))
            try:
                # Delete existing FAISS index files to force clean rebuild
                faiss_index_paths = [
                    os.path.join(settings.BASE_DIR, 'image_index.faiss'),
                    os.path.join(settings.BASE_DIR, 'text_index.faiss'),
                    os.path.join(settings.BASE_DIR, 'product_ids.npy')
                ]
                
                for path in faiss_index_paths:
                    if os.path.exists(path):
                        os.remove(path)
                        self.stdout.write(f"Removed existing index file: {path}")
                
                # Rebuild the indices
                image_index, text_index = build_faiss_index()
                
                if image_index is not None and text_index is not None:
                    self.stdout.write(self.style.SUCCESS(
                        f"FAISS indices rebuilt successfully. Image index: {image_index.ntotal} vectors, Text index: {text_index.ntotal} vectors"
                    ))
                else:
                    self.stderr.write(self.style.ERROR("Failed to rebuild FAISS indices."))
            except Exception as e:
                self.stderr.write(self.style.ERROR(f"Error rebuilding FAISS indices: {str(e)}"))
                traceback.print_exc()
        else:
            self.stdout.write(self.style.SUCCESS("No embeddings updated, skipping FAISS index rebuild."))