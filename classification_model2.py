import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json
from datetime import datetime
import time
import uuid
from tqdm import tqdm
from pymongo import MongoClient
from dotenv import load_dotenv
import os
#Model "https://huggingface.co/dangvantuan/vietnamese-embedding"
class DepartmentClassifierAPI:
    def __init__(self):
        # Load environment variables
        import os
        config_path = os.path.join(os.path.dirname(__file__), 'config.env')
        print(f"[DEBUG] Config path: {config_path}")
        print(f"[DEBUG] Config file exists: {os.path.exists(config_path)}")
        
        load_dotenv(config_path)
        
        # MongoDB configuration
        self.mongodb_uri = os.getenv('MONGODB_URI')
        self.mongodb_database = os.getenv('MONGODB_DATABASE')
        self.keywords_collection = os.getenv('MONGODB_KEYWORDS_COLLECTION')
        self.posts_collection = os.getenv('MONGODB_POSTS_COLLECTION')
        self.output_file = os.getenv('OUTPUT_FILE')
        
        # Debug: Print loaded values
        print(f"[DEBUG] MONGODB_URI loaded: {self.mongodb_uri is not None}")
        print(f"[DEBUG] MONGODB_DATABASE loaded: {self.mongodb_database}")
        print(f"[DEBUG] OUTPUT_FILE loaded: {self.output_file}")
        
        # Validate required environment variables
        if not self.mongodb_uri:
            raise ValueError("MONGODB_URI environment variable is required")
        if not self.mongodb_database:
            raise ValueError("MONGODB_DATABASE environment variable is required")
        if not self.keywords_collection:
            raise ValueError("MONGODB_KEYWORDS_COLLECTION environment variable is required")
        if not self.posts_collection:
            raise ValueError("MONGODB_POSTS_COLLECTION environment variable is required")
        if not self.output_file:
            raise ValueError("OUTPUT_FILE environment variable is required")
        
        # Connect to MongoDB
        self.client = MongoClient(self.mongodb_uri)
        self.db = self.client[self.mongodb_database]
        
        # Initialize Vietnamese embedding model
        self.model = SentenceTransformer('dangvantuan/vietnamese-embedding')
        
        self.departments = self._load_keywords()
        self.department_embeddings = self._calculate_department_embeddings()
        self.processed_items = set()
        self.results = []

    def _load_keywords(self):
        try:
            # Get keywords from MongoDB
            keywords_collection = self.db[self.keywords_collection]
            keywords_data = list(keywords_collection.find({}))
            
            if not keywords_data:
                print(f"[{uuid.uuid4()}] Warning: No keywords found in MongoDB")
                return {}
            
            departments = {}
            for item in keywords_data:
                if not isinstance(item, dict) or 'id_don_vi' not in item or 'tu_khoa' not in item:
                    print(f"[{uuid.uuid4()}] Warning: Invalid keyword item: {item}")
                    continue
                dept_id = item['id_don_vi']
                if dept_id not in departments:
                    departments[dept_id] = []
                departments[dept_id].append(item['tu_khoa'])
            
            print(f"[{uuid.uuid4()}] Loaded {len(departments)} departments with keywords from MongoDB")
            for dept_id, keywords in departments.items():
                print(f"[{uuid.uuid4()}] Department {dept_id}: {len(keywords)} keywords")
            return departments
        except Exception as e:
            print(f"[{uuid.uuid4()}] Error loading keywords from MongoDB: {str(e)}")
            return {}

    def _calculate_department_embeddings(self):
        department_embeddings = {}
        for dept_id, keywords in self.departments.items():
            dept_emb = self._get_embeddings(" ".join(keywords))
            department_embeddings[dept_id] = dept_emb.mean(axis=0)
        return department_embeddings

    def _get_embeddings(self, text):
        if not text or not isinstance(text, str):
            print(f"[{uuid.uuid4()}] Warning: Invalid text input for embeddings")
            return np.zeros((1, 768))  # Return zero vector with same dimension as vietnamese-embedding model
        
        try:
            # Truncate text if too long to avoid issues
            if len(text) > 512:
                text = text[:512]
            
            # Use SentenceTransformer to encode text
            embedding = self.model.encode(text)
            
            if embedding is None or len(embedding) == 0:
                print(f"[{uuid.uuid4()}] Warning: Empty embeddings generated")
                return np.zeros((1, 768))
            
            # Return embedding as 2D array for consistency with existing code
            return embedding.reshape(1, -1)
                    
        except Exception as e:
            print(f"[{uuid.uuid4()}] Error generating embeddings: {str(e)}")
            return np.zeros((1, 768))

    def classify_text(self, text):
        if not text or not isinstance(text, str):
            print(f"[{uuid.uuid4()}] Warning: Invalid text input for classification")
            return {}
            
        try:
            text_emb = self._get_embeddings(text)
            if text_emb is None or len(text_emb) == 0:
                print(f"[{uuid.uuid4()}] Warning: Empty embeddings for classification")
                return {}
                
            text_emb_mean = text_emb.mean(axis=0)
            
            results = {}
            for dept_id, dept_emb in self.department_embeddings.items():
                if dept_emb is None or len(dept_emb) == 0:
                    continue
                similarity = cosine_similarity([text_emb_mean], [dept_emb])[0][0]
                if similarity >= 0.6:
                    results[dept_id] = float(similarity * 100)
            return dict(sorted(results.items(), key=lambda x: x[1], reverse=True))
        except Exception as e:
            print(f"[{uuid.uuid4()}] Error in classification: {str(e)}")
            return {}

    def process_content(self):
        try:
            # Get posts from MongoDB
            posts_collection = self.db[self.posts_collection]
            posts = list(posts_collection.find({}))
            
            if not posts:
                print(f"[{uuid.uuid4()}] Warning: No posts found in MongoDB")
                return False
            
            print(f"[{uuid.uuid4()}] Found {len(posts)} posts in MongoDB")

            has_new_data = False
            for post in tqdm(posts, desc="Processing posts"):
                if not isinstance(post, dict) or 'postId' not in post:
                    print(f"[{uuid.uuid4()}] Warning: Invalid post format: {post}")
                    continue
                
                post_id = post['postId']
                if self._process_item(post, None):
                    has_new_data = True
        
            return has_new_data
        except Exception as e:
            print(f"[{uuid.uuid4()}] Error processing content from MongoDB: {str(e)}")
            return False

    def _process_item(self, post, comment):
        if not isinstance(post, dict) or 'postId' not in post:
            print(f"[{uuid.uuid4()}] Error: Invalid post data: {post}")
            return False
        
        try:
            post_id = post['postId']
            item_key = (post_id, "0")  # No comments for now

            if item_key in self.processed_items:
                return False
            
            # Get content from message or text field
            content = None
            if 'message' in post and post['message']:
                content = post['message']
            elif 'text' in post and post['text']:
                content = post['text']
            
            if not content:
                print(f"[{uuid.uuid4()}] Warning: Empty content for post {post_id}")
                return False
                
            classifications = self.classify_text(content)
            
            # Debug: Print classification results
            if classifications:
                print(f"[{uuid.uuid4()}] Post {post_id} classified to {len(classifications)} departments: {classifications}")
            else:
                print(f"[{uuid.uuid4()}] Post {post_id} not classified (no similarity >= 60%)")
            
            has_changes = False
            
            for dept_id, similarity in classifications.items():
                if similarity >= 60:
                    result_data = {
                        "postId": post_id,
                        "id_don_vi": dept_id,
                        "phan_tram_lien_quan": round(similarity, 2),
                        "content": content[:200] + "..." if len(content) > 200 else content,  # Truncate for readability
                        "timestamp": datetime.now().isoformat()
                    }
                    self.results.append(result_data)
                    print(f"[{uuid.uuid4()}] Success: Processed result for post {post_id}, department {dept_id}, similarity: {similarity:.2f}%")
                    has_changes = True
            
            if has_changes:
                self.processed_items.add(item_key)
            return has_changes
        except Exception as e:
            print(f"[{uuid.uuid4()}] Error processing item: {str(e)}")
            return False

    def save_results_to_json(self):
        """Lưu kết quả phân loại ra file JSON"""
        try:
            if not self.results:
                print(f"[{uuid.uuid4()}] Warning: No results to save")
                return False
            
            output_data = {
                "timestamp": datetime.now().isoformat(),
                "total_results": len(self.results),
                "results": self.results
            }
            
            with open(self.output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            
            print(f"[{uuid.uuid4()}] Successfully saved {len(self.results)} results to {self.output_file}")
            return True
        except Exception as e:
            print(f"[{uuid.uuid4()}] Error saving results to JSON: {str(e)}")
            return False

    def update_keywords(self):
        self.departments = self._load_keywords()
        self.department_embeddings = self._calculate_department_embeddings()
        return self.process_content()

def main():
    try:
        print(f"[{uuid.uuid4()}] Initializing Department Classifier with MongoDB...")
        classifier = DepartmentClassifierAPI()
        
        print(f"[{uuid.uuid4()}] Processing content from MongoDB...")
        has_new_data = classifier.process_content()
        
        if has_new_data:
            print(f"[{uuid.uuid4()}] Saving results to JSON file...")
            classifier.save_results_to_json()
            print(f"[{uuid.uuid4()}] Processing completed successfully!")
        else:
            print(f"[{uuid.uuid4()}] No new data to process.")
            print(f"[{uuid.uuid4()}] Total results collected: {len(classifier.results)}")
            if len(classifier.results) > 0:
                print(f"[{uuid.uuid4()}] Saving existing results to JSON file...")
                classifier.save_results_to_json()
            
        # Close MongoDB connection
        classifier.client.close()
        print(f"[{uuid.uuid4()}] MongoDB connection closed.")
            
    except Exception as e:
        print(f"[{uuid.uuid4()}] An error occurred: {str(e)}")

if __name__ == "__main__":
    main()