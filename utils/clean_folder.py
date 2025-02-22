import os
import sys

def clean_folder(folder_path):
    # Check if folder exists
    if not os.path.exists(folder_path):
        print(f"The folder {folder_path} doesn't exist!")
        return
    
    # List all non-jpg files
    files_to_remove = []
    for file in os.listdir(folder_path):
        full_path = os.path.join(folder_path, file)
        if not file.lower().endswith('.jpg') and os.path.isfile(full_path):
            files_to_remove.append(full_path)

    # Display files to be removed
    if files_to_remove:
        print("Files to be removed:")
        for file in files_to_remove:
            print(f"- {file}")
        
        confirm = input("\nDo you want to remove these files? (yes/no): ")
        if confirm.lower() == 'yes':
            for file in files_to_remove:
                try:
                    os.remove(file)
                    print(f"Removed: {file}")
                except Exception as e:
                    print(f"Error while removing {file}: {e}")
            print("\nCleanup completed!")
        else:
            print("Operation cancelled.")
    else:
        print("No files to remove!")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python clean_folder.py <folder_path>")
        sys.exit(1)
    folder = sys.argv[1]
    clean_folder(folder) 