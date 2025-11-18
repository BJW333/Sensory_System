#!/usr/bin/env python3
"""
Capture training faces - FIXED color space handling
"""
import cv2
import sys
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).parent))
from camera_manager import CameraManager
from deepface import DeepFace
import numpy as np

KNOWN_FACES_DIR = Path(__file__).parent / "knownfaces"

def main():
    print("="*60)
    print("TRAINING FACE CAPTURE TOOL (Color-Fixed)")
    print("="*60)
    
    name = input("\nEnter the person's name (e.g., 'soyam'): ").strip()
    if not name:
        print("❌ Name cannot be empty!")
        return
    
    person_dir = KNOWN_FACES_DIR / name
    person_dir.mkdir(parents=True, exist_ok=True)
    print(f"✅ Saving faces to: {person_dir}")
    
    print("\n📸 Instructions:")
    print("  - Look at the camera")
    print("  - Press SPACE to capture a face")
    print("  - Press 'q' to quit")
    print("\n🎥 Starting camera...\n")
    
    cam = CameraManager(0)
    time.sleep(1)
    
    capture_count = 0
    target_captures = 10
    
    while capture_count < target_captures:
        frame = cam.get_frame(wait=False)
        if frame is None:
            time.sleep(0.1)
            continue
        
        # Keep frame in BGR for display
        display_frame = frame.copy()
        
        # For face detection, convert to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        try:
            faces = DeepFace.extract_faces(
                img_path=frame_rgb,
                detector_backend="opencv",
                enforce_detection=False,
            )
        except:
            faces = []
        
        # Draw boxes
        for face in faces:
            r = face["facial_area"]
            x1, y1 = r["x"], r["y"]
            x2, y2 = x1 + r["w"], y1 + r["h"]
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(display_frame, "Press SPACE to save", 
                       (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Info overlay
        progress = f"Captured: {capture_count}/{target_captures}"
        cv2.putText(display_frame, progress, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(display_frame, "Press SPACE to capture, 'q' to quit", 
                   (10, frame.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow("Face Capture", display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("\n⏹️ Quitting...")
            break
        
        elif key == ord(' '):
            if len(faces) == 0:
                print("❌ No face detected!")
                continue
            
            if len(faces) > 1:
                print("⚠️ Multiple faces detected!")
                continue
            
            # Get face region from ORIGINAL BGR FRAME (not RGB)
            # This way we keep colors correct
            r = faces[0]["facial_area"]
            x1, y1 = r["x"], r["y"]
            x2, y2 = x1 + r["w"], y1 + r["h"]
            
            # Crop from BGR frame directly
            face_crop_bgr = frame[y1:y2, x1:x2]
            
            if face_crop_bgr.size == 0:
                print("❌ Face crop failed!")
                continue
            
            # Resize to 224x224
            face_crop_bgr = cv2.resize(face_crop_bgr, (224, 224))
            
            # Quality checks
            brightness = face_crop_bgr.mean()
            std_dev = face_crop_bgr.std()
            
            if brightness < 30:
                print(f"❌ Too dark (brightness: {brightness:.1f})")
                continue
            
            if std_dev < 15:
                print(f"❌ Too blurry (std: {std_dev:.1f})")
                continue
            
            # Save (cv2.imwrite expects BGR, which we have!)
            capture_count += 1
            filename = person_dir / f"{name}_{capture_count}_{int(time.time())}.jpg"
            success = cv2.imwrite(str(filename), face_crop_bgr)
            
            if success:
                print(f"✅ Captured {capture_count}/{target_captures}: {filename.name}")
                print(f"   Quality: brightness={brightness:.1f}, sharpness={std_dev:.1f}")
                
                # Show saved image briefly for verification
                preview = cv2.resize(face_crop_bgr, (400, 400))
                cv2.imshow("Saved Face Preview", preview)
                cv2.waitKey(500)  # Show for 0.5 seconds
                cv2.destroyWindow("Saved Face Preview")
            else:
                print(f"❌ Failed to save image!")
                capture_count -= 1
            
            time.sleep(0.3)
    
    cam.stop()
    cv2.destroyAllWindows()
    
    print("\n" + "="*60)
    print(f"✅ CAPTURE COMPLETE!")
    print(f"📁 Saved {capture_count} training images to: {person_dir}")
    print("="*60)
    
    if capture_count >= 5:
        print("\n✨ Ready for recognition!")
        print(f"   Run: python3 main.py")
    else:
        print(f"\n⚠️ Only {capture_count} images.")
        print("   Recommended: 5-10 images")

if __name__ == "__main__":
    main()