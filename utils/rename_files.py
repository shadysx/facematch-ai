import os
import sys

def rename_files(folder_path):
    # Check if folder exists
    if not os.path.exists(folder_path):
        print(f"The folder {folder_path} doesn't exist!")
        return
    
    # Get all jpg files
    jpg_files = [f for f in os.listdir(folder_path) 
                 if f.lower().endswith('.jpg') and os.path.isfile(os.path.join(folder_path, f))]
    
    if not jpg_files:
        print("No .jpg files found in the folder!")
        return
    
    # Sort files to ensure consistent ordering
    jpg_files.sort()
    
    # Preview the changes
    print("Files will be renamed as follows:")
    for i, old_name in enumerate(jpg_files, 1):
        new_name = f"face_{i:03d}.jpg"  # format: face_001.jpg, face_002.jpg, etc.
        print(f"{old_name} -> {new_name}")
    
    # Ask for confirmation
    confirm = input("\nDo you want to rename these files? (yes/no): ")
    if confirm.lower() == 'yes':
        # Rename files
        for i, old_name in enumerate(jpg_files, 1):
            old_path = os.path.join(folder_path, old_name)
            new_name = f"face_{i:03d}.jpg"
            new_path = os.path.join(folder_path, new_name)
            
            try:
                os.rename(old_path, new_path)
                print(f"Renamed: {old_name} -> {new_name}")
            except Exception as e:
                print(f"Error while renaming {old_name}: {e}")
        
        print("\nRenaming completed!")
    else:
        print("Operation cancelled.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python rename_files.py <folder_path>")
        sys.exit(1)
    folder = sys.argv[1]
    rename_files(folder) 