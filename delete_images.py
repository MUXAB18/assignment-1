import os
import random

def delete_random_images(folder_path, num_to_delete):
    """
    Delete a specified number of random images from a folder
    """
    # Get all files in the folder
    all_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    
    total_files = len(all_files)
    print(f"Total images in folder: {total_files}")
    
    if num_to_delete >= total_files:
        print(f"Cannot delete {num_to_delete} images. Only {total_files} images available.")
        return
    
    # Randomly select files to delete
    files_to_delete = random.sample(all_files, num_to_delete)
    
    print(f"Deleting {num_to_delete} random images...")
    
    deleted_count = 0
    for filename in files_to_delete:
        try:
            file_path = os.path.join(folder_path, filename)
            os.remove(file_path)
            deleted_count += 1
            
            # Progress indicator
            if deleted_count % 1000 == 0:
                print(f"Deleted {deleted_count}/{num_to_delete} images...")
        except Exception as e:
            print(f"Error deleting {filename}: {e}")
    
    remaining = len([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])
    print(f"\nDone! Deleted {deleted_count} images.")
    print(f"Remaining images: {remaining}")


if __name__ == "__main__":
    # Choose which folder to clean
    print("Which folder do you want to clean?")
    print("1. Data/Training/female")
    print("2. Data/Validation/female")
    print("3. Both")
    
    choice = input("Enter choice (1/2/3): ")
    
    if choice == "1":
        delete_random_images("Data/Training/male", 1000)
    elif choice == "2":
        delete_random_images("Data/Validation/male", 2000)
    elif choice == "3":
        delete_random_images("Data/Training/male", 15000)
        delete_random_images("Data/Validation/male", 2000)
    else:
        print("Invalid choice!")
