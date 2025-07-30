import cv2
import numpy as np
from attendance_backend import ArcFaceAttendanceBackend
import argparse

def main():
    backend = ArcFaceAttendanceBackend()
    
    parser = argparse.ArgumentParser(description='ArcFace Attendance System CLI')
    parser.add_argument('--register', type=str, help='Register person with name')
    parser.add_argument('--recognize', action='store_true', help='Start recognition mode')
    parser.add_argument('--list', action='store_true', help='List registered persons')
    parser.add_argument('--attendance', action='store_true', help='Show attendance records')
    
    args = parser.parse_args()
    
    if args.register:
        register_person(backend, args.register)
    elif args.recognize:
        recognize_mode(backend)
    elif args.list:
        list_persons(backend)
    elif args.attendance:
        show_attendance(backend)
    else:
        print("Use --help for available options")

def register_person(backend, name):
    print(f"Registering {name}...")
    print("Press SPACE to capture, ESC to exit")
    
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        cv2.imshow('Registration - Press SPACE to capture', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            success, message = backend.register_person(name, frame)
            print(f"Result: {message}")
            break
        elif key == 27:  # ESC
            break
    
    cap.release()
    cv2.destroyAllWindows()

def recognize_mode(backend):
    print("Recognition mode - Press ESC to exit")
    
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        name, distance = backend.recognize_person(frame)
        
        if name:
            cv2.putText(frame, f"{name} ({distance:.3f})", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Log attendance on recognition
            success, msg = backend.log_attendance(name)
            if success:
                print(f"Attendance logged: {msg}")
        else:
            cv2.putText(frame, "Unknown", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        cv2.imshow('Recognition - Press ESC to exit', frame)
        
        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break
    
    cap.release()
    cv2.destroyAllWindows()

def list_persons(backend):
    persons = backend.get_registered_persons()
    print(f"Registered persons ({len(persons)}):")
    for person in persons:
        print(f"  - {person}")

def show_attendance(backend):
    records = backend.get_attendance_records()
    print(f"Attendance records ({len(records)}):")
    for record in records:
        print(f"  {record['name']} - {record['timestamp']}")

if __name__ == "__main__":
    main()
