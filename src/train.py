import os
import sys
try:
    import torch
    torch_available = True
except ImportError:
    torch_available = False
    print("Warning: PyTorch not found. Training will run on CPU.")
import shutil
import glob
import random
from ultralytics import YOLO

def prepare_data(train_data_path):
    source_images_dir = os.path.join(train_data_path, 'images')
    source_labels_dir = os.path.join(train_data_path, 'labels')
    train_images_dir = os.path.join(train_data_path, 'train/images')
    train_labels_dir = os.path.join(train_data_path, 'train/labels')
    val_images_dir = os.path.join(train_data_path, 'val/images')
    val_labels_dir = os.path.join(train_data_path, 'val/labels')

    # Check if source directories exist
    if not os.path.isdir(source_images_dir):
        print(f"Error: Source images directory not found: {source_images_dir}")
        return False # Indicate failure
    if not os.path.isdir(source_labels_dir):
        print(f"Error: Source labels directory not found: {source_labels_dir}")
        return False # Indicate failure

    train_dir_exists = os.path.exists(train_images_dir) and os.path.exists(train_labels_dir)
    val_dir_exists = os.path.exists(val_images_dir) and os.path.exists(val_labels_dir)

    if train_dir_exists and val_dir_exists:
        print("Train and validation directories already exist. Verifying counts...")
        # Perform the count check
        if verify_counts(source_images_dir, train_images_dir, val_images_dir):
            print("Counts verified successfully. Skipping file preparation.")
            return True # Indicate success (pre-existing and correct)
        else:
            print("Count mismatch detected. Re-distributing files...")
            # Clean up existing train/val directories before re-running
            clean_up(train_data_path)
            # Proceed with the rest of the function to re-create and re-copy

    # Create train/val directories if they don't exist (or after cleanup)
    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(train_labels_dir, exist_ok=True)
    os.makedirs(val_images_dir, exist_ok=True)
    os.makedirs(val_labels_dir, exist_ok=True)

    # Find paired files in source directories
    paired_files = []
    try:
        image_files = {os.path.splitext(f)[0] for f in os.listdir(source_images_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp', '.gif'))}
        label_files = {os.path.splitext(f)[0] for f in os.listdir(source_labels_dir) if f.lower().endswith('.txt')}

        common_basenames = list(image_files.intersection(label_files))

        # Reconstruct full filenames (assuming jpg for images, txt for labels - adjust if needed)
        # Find the actual image extension
        img_ext_map = {}
        for f in os.listdir(source_images_dir):
             base, ext = os.path.splitext(f)
             if base in common_basenames and ext.lower() in ['.jpg', '.png', '.jpeg', '.bmp', '.gif']:
                 img_ext_map[base] = f

        paired_files = [(img_ext_map[base], base + '.txt') for base in common_basenames if base in img_ext_map]

        if not paired_files:
            print("Error: No matching image and label files found in source directories.")
            return False

    except FileNotFoundError as e:
        print(f"Error accessing source directories: {e}")
        return False
    except Exception as e:
        print(f"An error occurred while finding paired files: {e}")
        return False


    random.seed(42) # Use a fixed seed for reproducibility
    random.shuffle(paired_files)
    split_idx = int(len(paired_files) * 0.8)
    train_files = paired_files[:split_idx]
    val_files = paired_files[split_idx:]

    print(f"Total paired files: {len(paired_files)}, Training set: {len(train_files)}, Validation set: {len(val_files)}")

    # Copy files from source images/labels to train/val images/labels
    copy_files(train_files, source_images_dir, source_labels_dir, train_images_dir, train_labels_dir)
    copy_files(val_files, source_images_dir, source_labels_dir, val_images_dir, val_labels_dir)

    # Verify counts after copying (optional, but good for confirmation)
    print("Verifying counts after copying...")
    if not verify_counts(source_images_dir, train_images_dir, val_images_dir):
         print("Error: Count mismatch even after re-distribution.")
         return False # Indicate failure after re-distribution

    return True # Indicate success

def verify_counts(source_images_dir, train_images_dir, val_images_dir):
    """Checks if the number of images in train + val equals the source. Returns True if counts match, False otherwise."""
    match = False # Default to mismatch
    try:
        source_count = len([f for f in os.listdir(source_images_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp', '.gif'))])
        train_count = len([f for f in os.listdir(train_images_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp', '.gif'))]) if os.path.exists(train_images_dir) else 0
        val_count = len([f for f in os.listdir(val_images_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp', '.gif'))]) if os.path.exists(val_images_dir) else 0

        print(f"Source images count: {source_count}")
        print(f"Train images count: {train_count}")
        print(f"Validation images count: {val_count}")
        print(f"Total split images count: {train_count + val_count}")

        if source_count == (train_count + val_count):
            print("Verification successful: Total split image count matches source image count.")
            match = True
        else:
            print(f"Warning: Image count mismatch! Source: {source_count}, Split Total: {train_count + val_count}")
            match = False
    except FileNotFoundError as e:
        print(f"Error during count verification (directory not found): {e}")
    except Exception as e:
        print(f"An error occurred during count verification: {e}")
    return match


def move_files(files, base_path, data_type):
    # This function seems unused now and might need removal or adaptation
    # if it's intended for a different workflow.
    # For now, keeping it as is but noting it's not called by the modified prepare_data.
    for img_file, txt_file in files:
        src_img_path = os.path.join(base_path, img_file)
        dst_img_path = os.path.join(base_path, data_type, 'images', img_file)
        shutil.move(src_img_path, dst_img_path)

        src_txt_path = os.path.join(base_path, txt_file)
        dst_txt_path = os.path.join(base_path, data_type, 'labels', txt_file)
        shutil.move(src_txt_path, dst_txt_path)

def copy_files(files, src_img_dir, src_lbl_dir, dst_img_dir, dst_lbl_dir):
    """Copies specified image and label files from source to destination directories."""
    copied_count = 0
    for img_file, txt_file in files:
        src_img_path = os.path.join(src_img_dir, img_file)
        dst_img_path = os.path.join(dst_img_dir, img_file)
        src_txt_path = os.path.join(src_lbl_dir, txt_file)
        dst_txt_path = os.path.join(dst_lbl_dir, txt_file)

        try:
            if os.path.exists(src_img_path):
                 shutil.copy2(src_img_path, dst_img_path)
            else:
                 print(f"Warning: Source image not found, skipping copy: {src_img_path}")
                 continue # Skip this pair if image is missing

            if os.path.exists(src_txt_path):
                 shutil.copy2(src_txt_path, dst_txt_path)
            else:
                 print(f"Warning: Source label not found, skipping copy: {src_txt_path}")
                 # Decide if you want to remove the already copied image if label is missing
                 # os.remove(dst_img_path)
                 continue # Skip this pair if label is missing

            copied_count += 1
        except Exception as e:
            print(f"Error copying file pair ({img_file}, {txt_file}): {e}")
    print(f"Copied {copied_count} file pairs to {os.path.basename(dst_img_dir)}.")

def create_symlinks(files, base_path, data_type):
    # This function seems unused now and might need removal or adaptation.
    for img_file, txt_file in files:
        src_img_path = os.path.join(base_path, img_file)
        dst_img_path = os.path.join(base_path, data_type, 'images', img_file)
        os.symlink(src_img_path, dst_img_path)

        src_txt_path = os.path.join(base_path, txt_file)
        dst_txt_path = os.path.join(base_path, data_type, 'labels', txt_file)
        os.symlink(src_txt_path, dst_txt_path)

def clean_up(train_data_path):
    """Removes the train and val directories within the specified train_data_path."""
    print(f"Attempting to clean up train/val directories in {train_data_path}")
    train_dir = os.path.join(train_data_path, 'train')
    val_dir = os.path.join(train_data_path, 'val')
    if os.path.exists(train_dir):
        shutil.rmtree(train_dir, ignore_errors=True)
        print(f"Removed directory: {train_dir}")
    if os.path.exists(val_dir):
        shutil.rmtree(val_dir, ignore_errors=True)
        print(f"Removed directory: {val_dir}")

def copy_and_remove_latest_run_files(model_save_path, project_name):
    list_of_dirs = glob.glob('runs/detect/' + project_name + '*') # Add * to match project_name followed by a number
    if not list_of_dirs:
        print("No 'runs/detect/" + project_name + "' directories found. Skipping copy and removal.")
        return

    latest_dir = max(list_of_dirs, key=os.path.getmtime)

    if os.path.exists(latest_dir):
        else_destination_folder = os.path.join(model_save_path, 'else')
        os.makedirs(else_destination_folder, exist_ok=True)

        for item in os.listdir(latest_dir):
            s = os.path.join(latest_dir, item)
            
            if item == 'weights' and os.path.isdir(s):
                d = os.path.join(model_save_path, item)
                if os.path.isdir(s): # Should always be true for 'weights' if it exists as a dir
                    shutil.copytree(s, d, dirs_exist_ok=True)
                # else: # This case should ideally not happen if 'weights' is a file
                # shutil.copy2(s, d) 
            else:
                d = os.path.join(else_destination_folder, item)
                if os.path.isdir(s):
                    shutil.copytree(s, d, dirs_exist_ok=True)
                else:
                    shutil.copy2(s, d)
        best_pt_path = os.path.join(model_save_path, 'weights', 'best.pt')
        model = YOLO(best_pt_path)  # load a custom trained model

        # Export the model
        model.export(format="onnx")
        print(f"Exported model to ONNX format in {model_save_path}")
        
    runs_dir = 'runs'
    if os.path.exists(runs_dir) and os.path.isdir(runs_dir):
        shutil.rmtree(runs_dir)

def create_yaml(project_name, train_data_path, class_names, save_directory):
    # Ensure save_directory exists before creating yaml in it
    os.makedirs(save_directory, exist_ok=True) # Keep this check

    # Call prepare_data and check its return value
    if not prepare_data(train_data_path):
        print("Error during data preparation. Aborting YAML creation.")
        return None # Return None if preparation failed

    # Use os.path.abspath to ensure full paths are used, especially if train_data_path is relative
    # Point to the newly created/verified train and val directories
    train_path = os.path.abspath(os.path.join(train_data_path, 'train')).replace('\\', '/')
    val_path = os.path.abspath(os.path.join(train_data_path, 'val')).replace('\\', '/')

    # Remove leading spaces from lines 2, 3, and 4
    yaml_content = f"""train: {train_path}
val: {val_path}
nc: {len(class_names)}
names: [{', '.join(f"'{name}'" for name in class_names)}]
"""
    print(f"Project Name: {project_name}")
    # Use os.path.abspath for yaml_path as well
    yaml_path = os.path.abspath(os.path.join(save_directory, f'{project_name}.yaml')).replace('\\', '/')
    print(f"YAML Path: {yaml_path}")
    try: # Add error handling for file writing
        with open(yaml_path, 'w') as file:
            file.write(yaml_content)
        print(f"Successfully created YAML file: {yaml_path}") # Added success message
    except Exception as e:
        print(f"Error writing YAML file {yaml_path}: {e}")
        return None # Return None if YAML creation fails
    return yaml_path

def train_yolo(data_yaml, model_type, img_size, batch, epochs, model_save_path, project_name):
    # Ensure model_save_path exists before training potentially writes to it via copy_and_remove
    os.makedirs(model_save_path, exist_ok=True) # Keep this check

    # Determine device based on torch availability and CUDA status
    if torch_available and torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    print(f"Using device: {device}") # Added print statement for device
    try:
        model = YOLO(f'{model_type}.pt') # Load weights first
        if torch_available: # Only move model if torch is available
            model.to(device)
        print(f"Starting training: data={data_yaml}, epochs={epochs}, batch={batch}, imgsz={img_size}, name={project_name}")
        # Ensure project_name is passed correctly to avoid default 'train' folders
        # Add project='runs/detect' and exist_ok=True
        # Pass device explicitly to train method
        results = model.train(data=data_yaml, epochs=epochs, batch=batch, imgsz=img_size, project='runs/detect', name=project_name, save=True, exist_ok=True, device=device,workers=0)
        print("Training finished. Copying results...")
        copy_and_remove_latest_run_files(model_save_path, project_name)
        # Clean up train/val folders inside the original data path if they were created by prepare_data
        # Consider making clean_up optional or more specific
        # clean_up(os.path.dirname(data_yaml)) # This might delete the yaml file itself if it's in train_data_path
        print("Cleanup finished.")
    except Exception as e:
        print(f"An error occurred during training or cleanup: {e}")
        # Optionally re-raise the exception if needed: raise e
        results = None # Indicate failure
    return results

def parse_args():
    project_name = sys.argv[1]
    train_data_path = sys.argv[2]
    class_names = sys.argv[3].split(',')
    model_save_path = sys.argv[4]
    model_type = sys.argv[5]
    img_size = int(sys.argv[6])
    epochs = int(sys.argv[7])
    yaml_path = sys.argv[8]
    batch_size = int(sys.argv[9])

    results = train_yolo(yaml_path, model_type, img_size, batch_size, epochs, model_save_path, project_name)
    print(f"Training completed. Model saved to {model_save_path}")

if __name__ == '__main__':
    parse_args()