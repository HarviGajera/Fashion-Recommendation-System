import faiss
import numpy as np
import os
from django.conf import settings
from recommender.models import Product
from tqdm import tqdm
import logging
import traceback

# Set up logging
logger = logging.getLogger(__name__)

# Constants
IMAGE_INDEX_PATH = os.path.join(settings.BASE_DIR, 'image_index.faiss')
TEXT_INDEX_PATH = os.path.join(settings.BASE_DIR, 'text_index.faiss')
PRODUCT_IDS_PATH = os.path.join(settings.BASE_DIR, 'product_ids.npy')

def build_faiss_index():
    """Build FAISS indices for fast similarity search with improved error handling"""
    try:
        # Get all products from DB
        products = Product.objects.all()
        
        if not products.exists():
            logger.error("No products found in database")
            return None, None
        
        # Extract embeddings and ids
        image_embeddings = []
        text_embeddings = []
        product_ids = []
        
        logger.info(f"Building indices for {products.count()} products")
        
        # First pass: determine dimensions of embeddings
        first_product = products.first()
        if not first_product or not first_product.image_embedding or not first_product.text_embedding:
            logger.error("No valid products with embeddings found")
            return None, None
            
        # Get dimensions
        image_dim = len(np.frombuffer(first_product.image_embedding, dtype=np.float32))
        text_dim = len(np.frombuffer(first_product.text_embedding, dtype=np.float32))
        
        logger.info(f"Detected embedding dimensions - Image: {image_dim}, Text: {text_dim}")
        
        # Process all products
        for product in tqdm(products, desc="Processing product embeddings"):
            # Skip products with missing embeddings
            if not product.image_embedding or not product.text_embedding:
                logger.warning(f"Skipping product {product.id} with missing embeddings")
                continue
                
            try:
                # Convert binary embeddings to numpy arrays
                image_embedding = np.frombuffer(product.image_embedding, dtype=np.float32)
                text_embedding = np.frombuffer(product.text_embedding, dtype=np.float32)
                
                # Skip invalid embeddings
                if len(image_embedding) != image_dim:
                    logger.warning(f"Skipping product {product.id} with invalid image embedding dimension: {len(image_embedding)}")
                    continue
                    
                if len(text_embedding) != text_dim:
                    logger.warning(f"Skipping product {product.id} with invalid text embedding dimension: {len(text_embedding)}")
                    continue
                    
                # Normalize embeddings for cosine similarity
                image_norm = np.linalg.norm(image_embedding)
                text_norm = np.linalg.norm(text_embedding)
                
                if image_norm > 0:
                    image_embedding = image_embedding / image_norm
                else:
                    logger.warning(f"Zero norm for image embedding of product {product.id}")
                    continue
                
                if text_norm > 0:
                    text_embedding = text_embedding / text_norm
                else:
                    logger.warning(f"Zero norm for text embedding of product {product.id}")
                    continue
                
                # Add to collections
                image_embeddings.append(image_embedding)
                text_embeddings.append(text_embedding)
                product_ids.append(product.id)
                
            except Exception as e:
                logger.error(f"Error processing product {product.id}: {e}")
                traceback.print_exc()
        
        if not image_embeddings or not text_embeddings:
            logger.error("No valid embeddings found")
            return None, None
        
        # Convert lists to numpy arrays
        image_embeddings = np.array(image_embeddings).astype('float32')
        text_embeddings = np.array(text_embeddings).astype('float32')
        product_ids = np.array(product_ids)
        
        # Create and train indices
        # Using IndexFlatIP for Inner Product (cosine similarity with normalized vectors)
        image_index = faiss.IndexFlatIP(image_dim)
        text_index = faiss.IndexFlatIP(text_dim)
        
        # Add embeddings to indices
        image_index.add(image_embeddings)
        text_index.add(text_embeddings)
        
        # Save indices to disk
        try:
            faiss.write_index(image_index, IMAGE_INDEX_PATH)
            faiss.write_index(text_index, TEXT_INDEX_PATH)
            np.save(PRODUCT_IDS_PATH, product_ids)
            logger.info(f"Indices built successfully: {image_index.ntotal} vectors for images, {text_index.ntotal} vectors for text")
        except Exception as e:
            logger.error(f"Error saving indices: {e}")
            traceback.print_exc()
            return None, None
        
        return image_index, text_index
        
    except Exception as e:
        logger.error(f"Error building FAISS indices: {e}")
        traceback.print_exc()
        return None, None

def load_faiss_indices():
    """Load FAISS indices from disk with improved error handling"""
    try:
        # Check if all index files exist
        if not os.path.exists(IMAGE_INDEX_PATH) or not os.path.exists(TEXT_INDEX_PATH) or not os.path.exists(PRODUCT_IDS_PATH):
            logger.warning("One or more index files not found. Building indices...")
            image_index, text_index = build_faiss_index()
            if image_index is None or text_index is None:
                logger.error("Failed to build indices")
                return None, None, None
                
            # Load product IDs
            try:
                product_ids = np.load(PRODUCT_IDS_PATH)
                return image_index, text_index, product_ids
            except Exception as e:
                logger.error(f"Error loading product IDs: {e}")
                return None, None, None
        
        # Load indices from disk
        try:
            image_index = faiss.read_index(IMAGE_INDEX_PATH)
            text_index = faiss.read_index(TEXT_INDEX_PATH)
            product_ids = np.load(PRODUCT_IDS_PATH)
            
            # Verify indices are not empty
            if image_index.ntotal == 0 or text_index.ntotal == 0:
                logger.warning("One or more indices are empty. Rebuilding...")
                image_index, text_index = build_faiss_index()
                if image_index is None or text_index is None:
                    logger.error("Failed to rebuild indices")
                    return None, None, None
                    
                try:
                    product_ids = np.load(PRODUCT_IDS_PATH)
                except Exception as e:
                    logger.error(f"Error loading product IDs after rebuild: {e}")
                    return None, None, None
                
            logger.info(f"Indices loaded successfully: {image_index.ntotal} vectors for images, {text_index.ntotal} vectors for text")
            return image_index, text_index, product_ids
            
        except Exception as e:
            logger.error(f"Error reading indices from disk: {e}")
            logger.info("Attempting to rebuild indices...")
            
            # Try to remove corrupted files
            for path in [IMAGE_INDEX_PATH, TEXT_INDEX_PATH, PRODUCT_IDS_PATH]:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                        logger.info(f"Removed corrupted index: {path}")
                    except:
                        logger.warning(f"Could not remove file: {path}")
            
            # Build new indices
            image_index, text_index = build_faiss_index()
            if image_index is None or text_index is None:
                logger.error("Failed to rebuild indices")
                return None, None, None
                
            # Load product IDs if available
            try:
                product_ids = np.load(PRODUCT_IDS_PATH)
                return image_index, text_index, product_ids
            except:
                logger.error("Failed to load product IDs after rebuild")
                return None, None, None
        
    except Exception as e:
        logger.error(f"Error loading FAISS indices: {e}")
        traceback.print_exc()
        return None, None, None
        
def search_similar_items(query_embedding, index, k=5, max_results=10):
    """
    Search for similar items in the FAISS index with improved deduplication
    
    Parameters:
    - query_embedding: The embedding vector to search with
    - index: FAISS index for searching
    - k: Number of results to return after deduplication
    - max_results: Maximum number of results to fetch initially (to have extras for deduplication)
    
    Returns:
    - numpy array of unique product IDs
    """
    try:
        # Validate embedding
        if query_embedding is None:
            logger.error("Query embedding is None")
            return None
            
        if not isinstance(query_embedding, np.ndarray):
            logger.error(f"Query embedding is not a numpy array: {type(query_embedding)}")
            try:
                query_embedding = np.array(query_embedding, dtype=np.float32)
            except:
                return None
        
        # Get the index dimension
        index_dim = index.d
        embedding_dim = query_embedding.shape[0]
        
        # Check if dimensions match
        if embedding_dim != index_dim:
            logger.error(f"Embedding dimension mismatch: query is {embedding_dim}, index expects {index_dim}")
            return None
            
        # Ensure embedding is normalized for cosine similarity
        norm = np.linalg.norm(query_embedding)
        if norm > 0:
            query_embedding = query_embedding / norm
            
        # Reshape to (1, D) where D is embedding dimension
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        
        # Search the index with more results than needed to handle duplicates
        D, I = index.search(query_embedding, max_results)
        
        # Load product IDs
        try:
            product_ids = np.load(PRODUCT_IDS_PATH)
        except Exception as e:
            logger.error(f"Error loading product IDs: {e}")
            return None
        
        # Check if indices are valid
        if I is None or I.size == 0:
            logger.error("Search returned no results")
            return None
            
        if np.max(I) >= len(product_ids):
            logger.error(f"Search returned invalid indices: max {np.max(I)}, product_ids length {len(product_ids)}")
            return None
        
        # Map indices to product IDs
        all_result_ids = product_ids[I[0]]
        
        # Get unique product IDs (remove duplicates)
        unique_result_ids = np.unique(all_result_ids)
        
        # If after removing duplicates we have fewer than k results, just return what we have
        if len(unique_result_ids) <= k:
            return unique_result_ids
        
        # Otherwise return the top k unique results
        return unique_result_ids[:k]
        
    except Exception as e:
        logger.error(f"Error searching FAISS index: {e}")
        traceback.print_exc()
        return None