import cv2
import numpy as np
import pandas as pd
import os
import datetime
import logging
from typing import Tuple, Optional, List
import insightface
from insightface.app import FaceAnalysis

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ArcFaceAttendanceBackend:
    def __init__(self, threshold: float = 0.6):  # Lower threshold for better matching
        self.threshold = threshold
        self.face_data_csv = "face_data.csv"
        self.attendance_csv = "attendance.csv"
        
        # Enhanced ArcFace model with better detection
        self.app = FaceAnalysis(providers=['CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        
        # Face enhancement parameters
        self.min_face_size = 50
        self.max_faces = 10
        
        self._ensure_csv_files()
    
    def _ensure_csv_files(self):
        """Create CSV files if they don't exist"""
        try:
            # Check if we can write to current directory
            test_file = "test_write.tmp"
            with open(test_file, 'w') as f:
                f.write("test")
            os.remove(test_file)
            print("DEBUG: Directory is writable")
            
            if not os.path.exists(self.face_data_csv):
                df = pd.DataFrame(columns=['name', 'embedding'])
                df.to_csv(self.face_data_csv, index=False)
                print(f"DEBUG: Successfully created {self.face_data_csv}")
                
            if not os.path.exists(self.attendance_csv):
                df = pd.DataFrame(columns=['name', 'timestamp'])
                df.to_csv(self.attendance_csv, index=False)
                print(f"DEBUG: Successfully created {self.attendance_csv}")
                
        except PermissionError as e:
            print(f"DEBUG: Permission error: {e}")
        except Exception as e:
            print(f"DEBUG: Error creating CSV files: {e}")
    
    def enhance_image_quality(self, image: np.ndarray) -> np.ndarray:
        """Enhance image quality for better face detection"""
        # Convert to grayscale for processing
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Convert back to BGR if original was color
        if len(image.shape) == 3:
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        
        return enhanced
    
    def preprocess_image(self, image: np.ndarray) -> List[np.ndarray]:
        """Advanced preprocessing with multiple variations"""
        processed_images = []
        
        # Original image
        if len(image.shape) == 3 and image.shape[2] == 3:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = image
        processed_images.append(rgb_image)
        
        # Enhanced image
        enhanced = self.enhance_image_quality(image)
        if len(enhanced.shape) == 3:
            enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
        else:
            enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
        processed_images.append(enhanced_rgb)
        
        # Brightness variations
        for brightness in [0.8, 1.2]:
            bright_img = cv2.convertScaleAbs(rgb_image, alpha=brightness, beta=0)
            processed_images.append(bright_img)
        
        return processed_images
    
    def extract_face_embedding(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Enhanced face embedding extraction with multiple attempts"""
        try:
            processed_images = self.preprocess_image(image)
            all_embeddings = []
            
            for proc_img in processed_images:
                faces = self.app.get(proc_img)
                
                if len(faces) > 0:
                    # Get all valid faces
                    valid_faces = []
                    for face in faces:
                        bbox = face.bbox
                        face_width = bbox[2] - bbox[0]
                        face_height = bbox[3] - bbox[1]
                        
                        # Filter by minimum face size
                        if face_width >= self.min_face_size and face_height >= self.min_face_size:
                            valid_faces.append(face)
                    
                    if valid_faces:
                        # Get the largest face
                        largest_face = max(valid_faces, 
                                         key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
                        all_embeddings.append(largest_face.normed_embedding)
            
            if not all_embeddings:
                logger.warning("No face detected in any processed image")
                return None
            
            # Average multiple embeddings for robustness
            if len(all_embeddings) > 1:
                avg_embedding = np.mean(all_embeddings, axis=0)
                # Normalize the averaged embedding
                avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)
                return avg_embedding
            else:
                return all_embeddings[0]
                
        except Exception as e:
            logger.error(f"Error extracting embedding: {e}")
            return None
    
    def register_person(self, name: str, image: np.ndarray) -> Tuple[bool, str]:
        """Enhanced registration with multiple embeddings"""
        try:
            if self.person_exists(name):
                return False, f"Person '{name}' already registered"
            
            # Extract multiple embeddings for robustness
            embeddings = []
            processed_images = self.preprocess_image(image)
            
            for proc_img in processed_images:
                embedding = self.extract_single_embedding(proc_img)
                if embedding is not None:
                    embeddings.append(embedding)
            
            if not embeddings:
                return False, "No face detected in image"
            
            # Store multiple embeddings (up to 3 best ones)
            best_embeddings = embeddings[:3]
            
            for i, embedding in enumerate(best_embeddings):
                embedding_str = ','.join(map(str, embedding))
                new_row = pd.DataFrame({
                    'name': [f"{name}_{i}" if i > 0 else name],
                    'embedding': [embedding_str]
                })
                
                df = pd.read_csv(self.face_data_csv)
                df = pd.concat([df, new_row], ignore_index=True)
                df.to_csv(self.face_data_csv, index=False)
            
            logger.info(f"Successfully registered {name} with {len(best_embeddings)} embeddings")
            return True, f"Successfully registered {name} with enhanced recognition"
            
        except Exception as e:
            logger.error(f"Error registering person: {e}")
            return False, f"Registration failed: {str(e)}"
    
    def extract_single_embedding(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Extract single embedding from preprocessed image"""
        try:
            faces = self.app.get(image)
            if len(faces) == 0:
                return None
            
            # Get the largest face
            face = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
            return face.normed_embedding
            
        except Exception as e:
            return None
    
    def person_exists(self, name: str) -> bool:
        """Check if person is already registered"""
        try:
            df = pd.read_csv(self.face_data_csv)
            return name in df['name'].values
        except:
            return False
    
    def recognize_person(self, image: np.ndarray) -> Tuple[Optional[str], float]:
        """Enhanced recognition with multiple matching strategies"""
        try:
            # Extract embedding from test image
            test_embedding = self.extract_face_embedding(image)
            if test_embedding is None:
                return None, 0.0
            
            # Load stored embeddings
            df = pd.read_csv(self.face_data_csv)
            if df.empty:
                return None, 0.0
            
            best_matches = []
            
            for _, row in df.iterrows():
                stored_embedding = np.array([float(x) for x in row['embedding'].split(',')])
                
                # Calculate multiple distance metrics
                euclidean_dist = np.linalg.norm(test_embedding - stored_embedding)
                cosine_sim = np.dot(test_embedding, stored_embedding)
                
                # Combined score (lower is better)
                combined_score = euclidean_dist - cosine_sim
                
                name = row['name'].split('_')[0]  # Remove suffix if exists
                best_matches.append((name, combined_score, euclidean_dist, cosine_sim))
            
            # Group by person name and find best match per person
            person_scores = {}
            for name, score, eucl, cos in best_matches:
                if name not in person_scores or score < person_scores[name][0]:
                    person_scores[name] = (score, eucl, cos)
            
            if not person_scores:
                return None, 0.0
            
            # Find overall best match
            best_person = min(person_scores.items(), key=lambda x: x[1][0])
            best_name = best_person[0]
            best_euclidean = best_person[1][1]
            
            # More lenient threshold check
            if best_euclidean <= self.threshold:
                logger.info(f"Recognized {best_name} with distance {best_euclidean:.3f}")
                return best_name, best_euclidean
            else:
                logger.info(f"No match found. Best distance: {best_euclidean:.3f}")
                return None, best_euclidean
                
        except Exception as e:
            logger.error(f"Error recognizing person: {e}")
            return None, 0.0
    
    def log_attendance(self, name: str) -> Tuple[bool, str]:
        """Log attendance for recognized person"""
        try:
            # Ensure attendance.csv exists with proper structure
            if not os.path.exists(self.attendance_csv):
                df = pd.DataFrame(columns=['name', 'timestamp'])
                df.to_csv(self.attendance_csv, index=False)
            
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')  # Remove seconds
            today = datetime.datetime.now().strftime('%Y-%m-%d')
            
            # Read existing records
            try:
                df = pd.read_csv(self.attendance_csv)
                # Clean any empty rows or NaN values
                df = df.dropna()
            except (pd.errors.EmptyDataError, FileNotFoundError):
                df = pd.DataFrame(columns=['name', 'timestamp'])
            
            # Check if already logged today (safer approach)
            if not df.empty:
                # Convert to string and check for today's date
                df_today = df[df['timestamp'].astype(str).str.startswith(today)]
                if name in df_today['name'].values:
                    return False, f"Attendance already logged for {name} today"
            
            # Add new record
            new_row = pd.DataFrame({
                'name': [name],
                'timestamp': [timestamp]
            })
            
            df = pd.concat([df, new_row], ignore_index=True)
            df.to_csv(self.attendance_csv, index=False)
            
            logger.info(f"Logged attendance for {name} at {timestamp}")
            return True, f"Attendance logged for {name} at {timestamp}"
        
        except Exception as e:
            logger.error(f"Error logging attendance: {e}")
            return False, f"Failed to log attendance: {str(e)}"
    
    def get_registered_persons(self) -> List[str]:
        """Get list of registered persons"""
        try:
            df = pd.read_csv(self.face_data_csv)
            return df['name'].tolist()
        except:
            return []
    
    def get_attendance_records(self) -> List[dict]:
        try:
            df = pd.read_csv(self.attendance_csv)
            print(f"DEBUG: CSV columns: {df.columns.tolist()}")  # Check columns
            print(f"DEBUG: Sample records: {df.head().to_dict('records')}")  # Check data
            return df.to_dict('records')
        except Exception as e:
            print(f"DEBUG: Error reading CSV: {e}")
            return []
    
    def register_person_multiple_angles(self, name: str, images: List[np.ndarray]) -> Tuple[bool, str]:
        """Register person with multiple angle images"""
        try:
            if self.person_exists(name):
                return False, f"Person '{name}' already registered"
            
            all_embeddings = []
            
            for i, image in enumerate(images):
                embedding = self.extract_face_embedding(image)
                if embedding is not None:
                    all_embeddings.append(embedding)
            
            if len(all_embeddings) < 1:
                return False, "No faces detected in provided images"
            
            # Store up to 5 best embeddings
            for i, embedding in enumerate(all_embeddings[:5]):
                embedding_str = ','.join(map(str, embedding))
                new_row = pd.DataFrame({
                    'name': [f"{name}_{i}" if i > 0 else name],
                    'embedding': [embedding_str]
                })
                
                df = pd.read_csv(self.face_data_csv)
                df = pd.concat([df, new_row], ignore_index=True)
                df.to_csv(self.face_data_csv, index=False)
            
            return True, f"Registered {name} with {len(all_embeddings)} angle variations"
        
        except Exception as e:
            return False, f"Registration failed: {str(e)}"
