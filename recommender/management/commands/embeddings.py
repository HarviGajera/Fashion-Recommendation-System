from django.core.management.base import BaseCommand
from django.conf import settings
import os
from recommender.utils import process_csv_and_store_data
from recommender.models import Product
import numpy as np
from tqdm import tqdm
from recommender.faiss_index import build_faiss_index

class Command(BaseCommand):
    help = 'Download images, generate CLIP & SentenceTransformer embeddings, train alignment model, and store in DB'

    def add_arguments(self, parser):
        parser.add_argument('--csv', type=str, help='Path to CSV file', default='fashion_data.csv')
        parser.add_argument('--batch-size', type=int, help='Batch size for DB operations', default=50)
        parser.add_argument('--limit', type=int, help='Limit number of products to process', default=None)
        parser.add_argument('--model', type=str, help='SentenceTransformer model to use', 
                            default='all-MiniLM-L6-v2', 
                            choices=['all-MiniLM-L6-v2', 'all-mpnet-base-v2', 'paraphrase-multilingual-MiniLM-L12-v2'])

    def handle(self, *args, **kwargs):
        csv_filename = kwargs.get('csv')
        batch_size = kwargs.get('batch_size')
        limit = kwargs.get('limit')
        
        csv_path = os.path.join(settings.BASE_DIR, csv_filename)
        if not os.path.exists(csv_path):
            self.stderr.write(self.style.ERROR(f"CSV file not found: {csv_path}"))
            return
        
        media_folder = os.path.join(settings.MEDIA_ROOT, 'products')
        os.makedirs(media_folder, exist_ok=True)

        # Process the CSV and get valid rows with generated embeddings
        self.stdout.write(self.style.SUCCESS("Starting data processing..."))
        valid_rows = process_csv_and_store_data(csv_path, media_folder)
        
        if limit:
            valid_rows = valid_rows[:limit]
            self.stdout.write(f"Limited to {limit} products")

        # Batch save to database
        self.stdout.write(self.style.SUCCESS("Saving products to database..."))
        
        for i in range(0, len(valid_rows), batch_size):
            batch = valid_rows[i:i+batch_size]
            products_to_create = []
            
            for item in tqdm(batch, desc=f"Preparing batch {i//batch_size + 1}/{(len(valid_rows)//batch_size) + 1}"):
                # Convert embeddings to binary format for storage
                image_binary = np.array(item['image_embedding'], dtype=np.float32).tobytes()
                text_binary = np.array(item['text_embedding'], dtype=np.float32).tobytes()
                
                products_to_create.append(
                    Product(
                        product_name=item['product_name'],
                        category=item['category'],
                        image='products/' + os.path.basename(item['image']),
                        image_embedding=image_binary,
                        text_embedding=text_binary
                    )
                )
            
            # Bulk create the products
            Product.objects.bulk_create(products_to_create)
            self.stdout.write(f"Saved batch {i//batch_size + 1}/{(len(valid_rows)//batch_size) + 1} to database")

        self.stdout.write(self.style.SUCCESS(f"Successfully processed {len(valid_rows)} items and saved to DB."))
        
        # Build FAISS indices
        self.stdout.write(self.style.SUCCESS("Building FAISS indices..."))
        image_index, text_index = build_faiss_index()
        
        if image_index is not None and text_index is not None:
            self.stdout.write(self.style.SUCCESS("FAISS indices built successfully."))
        else:
            self.stderr.write(self.style.ERROR("Failed to build FAISS indices."))