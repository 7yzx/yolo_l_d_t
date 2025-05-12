import shutil
import os
import glob
from ultralytics import YOLO
import traceback # Import traceback for detailed error logging

def move_detection_results(source_dir, target_dir):
    try:
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        # Ensure source exists before iterating
        if not os.path.isdir(source_dir):
             print(f"Error: Source directory {source_dir} not found for moving results.")
             return False # Indicate failure

        # Remove existing target directory content if it exists
        if os.path.exists(target_dir):
            print(f"Removing existing target directory: {target_dir}")
            shutil.rmtree(target_dir)
            os.makedirs(target_dir) # Recreate empty target directory

        # Move contents
        for item in os.listdir(source_dir):
            s = os.path.join(source_dir, item)
            d = os.path.join(target_dir, item)
            shutil.move(s, d) # Move items individually

        # Remove the now empty source directory
        shutil.rmtree(source_dir)
        print(f"Successfully moved results from {source_dir} to {target_dir}")
        return True # Indicate success
    except Exception as e:
        print(f"Error moving detection results from {source_dir} to {target_dir}: {e}")
        traceback.print_exc() # Print detailed traceback
        return False # Indicate failure


def detect_images(images_folder, model_path, callback=None):
    try:
        print(f"Detecting images in: {images_folder}")
        print(f"Using model: {model_path}")
        model = YOLO(model_path)

        # Define a predictable temporary run name
        temp_run_name = 'temp_predict_run'
        run_project_dir = 'runs/detect'
        source_run_dir = os.path.join(run_project_dir, temp_run_name)

        # Run prediction
        results = model.predict(
            source=images_folder,
            save=True,
            save_txt=True,
            imgsz=640,
            conf=0.5,
            project=run_project_dir,
            name=temp_run_name,
            exist_ok=True # Overwrite the temp run dir if it exists
        )
        print(f"Prediction finished. Raw results in: {source_run_dir}")

        # Define final results directory
        results_dir = os.path.join(images_folder, 'detection_results')

        # Move results and check for success
        if move_detection_results(source_run_dir, results_dir):
            # Optional: Clean up parent 'runs/detect' if empty
            try:
                if not os.listdir(run_project_dir):
                    os.rmdir(run_project_dir)
                if not os.listdir(os.path.dirname(run_project_dir)): # Check 'runs' dir
                     os.rmdir(os.path.dirname(run_project_dir))
            except OSError as e:
                 print(f"Note: Could not remove empty run directories: {e}")

            if callback:
                callback(results_dir)
            return results_dir # Return path on success
        else:
            print("Failed to move detection results.")
            return None # Return None on failure during move

    except FileNotFoundError as e:
        print(f"Error during detection (File Not Found): {e}")
        traceback.print_exc()
        return None # Return None on failure
    except Exception as e:
        print(f"An unexpected error occurred in detect_images: {e}")
        traceback.print_exc() # Print detailed traceback
        return None # Return None on failure
    
if __name__ == '__main__':
    print("Starting detection process...")
    