import cv2
import threading
import time
import os
from ultralytics import YOLO
try:
    import torch
    torch_available = True
except ImportError:
    torch_available = False
    print("Warning: PyTorch not found. Camera detection will run on CPU.")
from datetime import datetime

class CameraDetection:
    def __init__(self, model_path, conf_threshold=0.5, frame_update_callback=None):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold

        # Determine device based on torch availability and CUDA status
        if torch_available and torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        print(f"CameraDetection using device: {self.device}")

        if torch_available: # Only move model if torch is available
            self.model.to(self.device)

        self.cap = None
        self.running = False
        self.save_dir = ""
        self.scene_id = 0
        self.frame_update_callback = frame_update_callback # Store the callback

    def start_camera(self, camera_id):
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise ValueError("Unable to open camera")

        self.original_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.original_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def stop_camera(self):
        if self.cap:
            self.cap.release()
            self.cap = None

    def set_save_directory(self, directory):
        self.save_dir = directory
        os.makedirs(self.save_dir, exist_ok=True) # Ensure directory exists

    def show_camera_stream(self, frame_update_callback=None):
        if frame_update_callback:
            self.frame_update_callback = frame_update_callback
        if not self.frame_update_callback:
            print("Warning: No frame update callback provided to show_camera_stream.")
            # Decide if you want to raise an error or just print a warning
            # raise ValueError("Frame update callback is required to show stream.")
            return # Or simply don't start the thread if no callback

        self.running = True
        # Pass self.frame_update_callback to the thread
        threading.Thread(target=self._update_stream, daemon=True).start()

    def _update_stream(self):
        while self.running:
            if not self.cap or not self.cap.isOpened():
                print("Camera not opened or released.")
                self.running = False
                break

            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame.")
                # Optionally add a small delay or attempt reconnect
                time.sleep(0.1)
                continue # Try again or break

            # Perform detection
            # Pass device explicitly to model call
            results = self.model(frame, verbose=False, device=self.device)
            detection_frame = frame.copy()
            self._draw_bounding_boxes(detection_frame, results)

            # Convert to RGB for consistency if needed by callback
            img_rgb = cv2.cvtColor(detection_frame, cv2.COLOR_BGR2RGB)

            # Call the callback with the processed frame (NumPy array)
            if self.frame_update_callback:
                try:
                    self.frame_update_callback(img_rgb)
                except Exception as e:
                    print(f"Error in frame update callback: {e}")
                    # Consider stopping or logging more details

            # Control frame rate
            time.sleep(0.03) # ~30 FPS target

        print("Exiting camera update stream loop.")

    def _draw_bounding_boxes(self, frame, results):
        colors = {}
        # Check if results and boxes exist and are not empty
        if results and results[0].boxes and len(results[0].boxes) > 0:
            for result in results[0].boxes:
                if result.conf is not None and len(result.conf) > 0 and result.conf[0] >= self.conf_threshold:
                    if result.xyxy is None or len(result.xyxy) == 0: continue
                    x1, y1, x2, y2 = map(int, result.xyxy[0])

                    if result.cls is None or len(result.cls) == 0: continue
                    class_index = int(result.cls[0])

                    # Ensure class_index is valid for model.names
                    if self.model.names and 0 <= class_index < len(self.model.names):
                        label = self.model.names[class_index]
                    else:
                        label = f"Class_{class_index}" # Fallback label

                    confidence = result.conf[0]

                    # Generate color for the label if not already present
                    if label not in colors:
                        # Simple hash-based color generation
                        hash_val = hash(label)
                        colors[label] = (hash_val % 256, (hash_val * 7) % 256, (hash_val * 13) % 256)

                    color = colors[label]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    text = f"{label}: {confidence:.2f}"
                    # Put text slightly above the box, ensuring it stays within frame boundaries
                    text_y = y1 - 10 if y1 - 10 > 10 else y1 + 15
                    cv2.putText(frame, text, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        # else:
            # Optional: print a message if no detections or results structure is unexpected
            # print("No detections in this frame or results format issue.")

    def capture_frame(self):
        # Ensure camera is running and frame can be read
        if not self.running or not self.cap or not self.cap.isOpened():
            print("Camera not running or not available for capture.")
            return None, None, None # Indicate failure

        ret, frame = self.cap.read()
        if not ret or frame is None:
            print("Capture failed: Could not read frame.")
            return None, None, None # Indicate failure

        # Ensure save directory exists
        if not self.save_dir:
            print("Error: Save directory not set.")
            # Optionally set a default directory here or raise an error
            self.save_dir = "." # Example: default to current directory
            os.makedirs(self.save_dir, exist_ok=True)
            print(f"Warning: Save directory was not set. Defaulting to: {os.path.abspath(self.save_dir)}")
        elif not os.path.exists(self.save_dir):
             try:
                 os.makedirs(self.save_dir, exist_ok=True)
             except OSError as e:
                 print(f"Error creating save directory '{self.save_dir}': {e}")
                 return None, None, None # Indicate failure


        self.scene_id += 1
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        base_filename = f"{timestamp}_{self.scene_id:04d}"

        origin_image_path = os.path.join(self.save_dir, f"{base_filename}_origin.jpg")
        try:
            cv2.imwrite(origin_image_path, frame)
        except Exception as e:
            print(f"Error saving original image to {origin_image_path}: {e}")
            return None, None, None # Indicate failure

        # Perform detection on the captured frame
        detection_frame = frame.copy()
        try:
            # Pass device explicitly to model call
            results = self.model(detection_frame, verbose=False, device=self.device) # Use the instance model
            self._draw_bounding_boxes(detection_frame, results) # Use the instance method
        except Exception as e:
            print(f"Error during detection for captured frame: {e}")
            # Decide if you want to save the detection image/txt anyway or return failure
            # Saving without boxes might be an option, or return failure:
            return origin_image_path, None, None # Indicate partial success/failure

        detection_image_path = os.path.join(self.save_dir, f"{base_filename}_detection.jpg")
        try:
            cv2.imwrite(detection_image_path, detection_frame)
        except Exception as e:
            print(f"Error saving detection image to {detection_image_path}: {e}")
            # Return paths obtained so far, indicating detection image save failed
            return origin_image_path, None, None

        txt_path = os.path.join(self.save_dir, f"{base_filename}_detection.txt")
        try:
            with open(txt_path, 'w') as f:
                # Ensure results and boxes exist before iterating
                if results and results[0].boxes and len(results[0].boxes) > 0:
                    # Get original frame dimensions if available, otherwise use current frame size
                    h, w = frame.shape[:2]
                    original_width = self.original_width if hasattr(self, 'original_width') else w
                    original_height = self.original_height if hasattr(self, 'original_height') else h
                    if original_width == 0 or original_height == 0: # Fallback if attributes weren't set
                        original_width, original_height = w, h

                    for result in results[0].boxes:
                         if result.conf is not None and len(result.conf) > 0 and result.conf[0] >= self.conf_threshold:
                            if result.xyxy is None or len(result.xyxy) == 0: continue
                            x1, y1, x2, y2 = map(int, result.xyxy[0])

                            if result.cls is None or len(result.cls) == 0: continue
                            class_index = int(result.cls[0])

                            # Normalize coordinates using original dimensions
                            x_center = (x1 + x2) / (2 * original_width)
                            y_center = (y1 + y2) / (2 * original_height)
                            width_norm = (x2 - x1) / original_width
                            height_norm = (y2 - y1) / original_height
                            # Clamp values to [0.0, 1.0] just in case
                            x_center = max(0.0, min(1.0, x_center))
                            y_center = max(0.0, min(1.0, y_center))
                            width_norm = max(0.0, min(1.0, width_norm))
                            height_norm = max(0.0, min(1.0, height_norm))

                            f.write(f"{class_index} {x_center:.6f} {y_center:.6f} {width_norm:.6f} {height_norm:.6f}\n")
        except Exception as e:
            print(f"Error writing detection labels to {txt_path}: {e}")
            # Return paths obtained so far, indicating label writing failed
            return origin_image_path, detection_image_path, None

        return origin_image_path, detection_image_path, txt_path

    def stop(self):
        print("Stopping camera detection...")
        self.running = False
        # Add a small delay to allow the _update_stream loop to exit
        time.sleep(0.2)
        self.stop_camera()
        print("Camera stopped.")