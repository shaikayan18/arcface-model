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
    def __init__(self, threshold: float = 0.8):
        self.threshold = threshold
        self.face_data_csv = "face_data.csv"
        self.attendance_csv = "attendance.csv"
        self.ip_tracking_csv = "ip_tracking.csv"  # New CSV for IP tracking
        
        # Initialize ArcFace model
        self.app = FaceAnalysis(providers=['CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        
        # Ensure CSV files exist
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
                
            if not os.path.exists(self.ip_tracking_csv):
                df = pd.DataFrame(columns=['name', 'ip_address', 'timestamp', 'action'])
                df.to_csv(self.ip_tracking_csv, index=False)
                print(f"DEBUG: Successfully created {self.ip_tracking_csv}")
                
        except PermissionError as e:
            print(f"DEBUG: Permission error: {e}")
        except Exception as e:
            print(f"DEBUG: Error creating CSV files: {e}")
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for ArcFace model"""
        # Convert BGR to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    
    def extract_face_embedding(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Extract face embedding using ArcFace"""
        try:
            image = self.preprocess_image(image)
            faces = self.app.get(image)
            
            if len(faces) == 0:
                logger.warning("No face detected in image")
                return None
                
            # Get the largest face (most prominent)
            face = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
            
            # Normalize embedding
            embedding = face.normed_embedding
            return embedding
            
        except Exception as e:
            logger.error(f"Error extracting embedding: {e}")
            return None
    
    def register_person(self, name: str, image: np.ndarray) -> Tuple[bool, str]:
        """Register a person with their face embedding"""
        try:
            # Check if person already exists
            if self.person_exists(name):
                return False, f"Person '{name}' already registered"
            
            # Extract embedding
            embedding = self.extract_face_embedding(image)
            if embedding is None:
                return False, "No face detected in image"
            
            # Save to CSV
            embedding_str = ','.join(map(str, embedding))
            new_row = pd.DataFrame({
                'name': [name],
                'embedding': [embedding_str]
            })
            
            df = pd.read_csv(self.face_data_csv)
            df = pd.concat([df, new_row], ignore_index=True)
            df.to_csv(self.face_data_csv, index=False)
            
            logger.info(f"Successfully registered {name}")
            return True, f"Successfully registered {name}"
            
        except Exception as e:
            logger.error(f"Error registering person: {e}")
            return False, f"Registration failed: {str(e)}"
    
    def person_exists(self, name: str) -> bool:
        """Check if person is already registered"""
        try:
            df = pd.read_csv(self.face_data_csv)
            return name in df['name'].values
        except:
            return False
    
    def recognize_person(self, image: np.ndarray) -> Tuple[Optional[str], float]:
        """Recognize person from image using stored embeddings"""
        try:
            # Extract embedding from test image
            test_embedding = self.extract_face_embedding(image)
            if test_embedding is None:
                return None, 0.0
            
            # Load stored embeddings
            df = pd.read_csv(self.face_data_csv)
            if df.empty:
                return None, 0.0
            
            best_match = None
            best_distance = float('inf')
            
            for _, row in df.iterrows():
                stored_embedding = np.array([float(x) for x in row['embedding'].split(',')])
                
                # Calculate Euclidean distance
                distance = np.linalg.norm(test_embedding - stored_embedding)
                
                if distance < best_distance:
                    best_distance = distance
                    best_match = row['name']
            
            # Check if distance is within threshold
            if best_distance <= self.threshold:
                logger.info(f"Recognized {best_match} with distance {best_distance:.3f}")
                return best_match, best_distance
            else:
                logger.info(f"No match found. Best distance: {best_distance:.3f}")
                return None, best_distance
                
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
        """Get attendance records with proper error handling"""
        try:
            # Check if file exists
            if not os.path.exists(self.attendance_csv):
                logger.info("Attendance CSV file not found, returning empty records")
                return []
            
            # Read CSV file
            df = pd.read_csv(self.attendance_csv)
            
            # Handle empty file
            if df.empty:
                logger.info("Attendance CSV file is empty")
                return []
            
            # Clean data - remove any rows with NaN values
            df = df.dropna()
            
            # Ensure columns exist
            if 'name' not in df.columns or 'timestamp' not in df.columns:
                logger.error("Invalid CSV structure - missing required columns")
                return []
            
            # Convert to records and return
            records = df.to_dict('records')
            logger.info(f"Successfully loaded {len(records)} attendance records")
            return records
            
        except pd.errors.EmptyDataError:
            logger.info("Attendance CSV file is empty")
            return []
        except Exception as e:
            logger.error(f"Error reading attendance records: {e}")
            return []

    def log_ip_activity(self, name: str, ip_address: str, action: str) -> bool:
        """Log IP activity for users"""
        try:
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Create new record
            new_row = pd.DataFrame({
                'name': [name],
                'ip_address': [ip_address],
                'timestamp': [timestamp],
                'action': [action]  # 'registration', 'recognition', 'attendance'
            })
            
            # Read existing records and append
            try:
                df = pd.read_csv(self.ip_tracking_csv)
            except (pd.errors.EmptyDataError, FileNotFoundError):
                df = pd.DataFrame(columns=['name', 'ip_address', 'timestamp', 'action'])
            
            df = pd.concat([df, new_row], ignore_index=True)
            df.to_csv(self.ip_tracking_csv, index=False)
            
            logger.info(f"Logged IP activity: {name} from {ip_address} - {action}")
            return True
            
        except Exception as e:
            logger.error(f"Error logging IP activity: {e}")
            return False

    def get_ip_records(self) -> List[dict]:
        """Get IP tracking records"""
        try:
            df = pd.read_csv(self.ip_tracking_csv)
            return df.to_dict('records')
        except:
            return []
