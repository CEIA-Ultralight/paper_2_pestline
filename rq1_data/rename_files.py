import os
import sys


def rename_images(folder_path:str = None, dry_run:bool = False):
    # --- CONFIGURATION ---
    # 1. Paste the path to your folder here.
    # Windows example: r"C:\Users\Name\Documents\Images"
    # Mac/Linux example: "/Users/Name/Documents/Images"
    folder_path = folder_path if folder_path else os.getcwd()

    # 2. Set dry_run to False when you are ready to actually rename the files.
    # True = Just print what would happen (Safety mode).
    # False = Actually rename the files.
    # ---------------------

    # Verify the folder exists
    if not os.path.exists(folder_path):
        print(f"Error: The folder '{folder_path}' does not exist.")
        return

    print(f"Scanning folder: {folder_path}")
    print("-" * 50)

    count = 0

    # Loop through all files in the directory
    for filename in os.listdir(folder_path):

        # Check if the file fits the pattern containing "_jpg"
        if "_jpg" in filename and (filename.endswith(".jpg") or filename.endswith(".txt")):

            # Logic: Split the string at "_jpg"
            # Example: "20251106_154953_jpg.rf.hash.jpg"
            # Becomes: ["20251106_154953", ".rf.hash.jpg"]
            parts = filename.split("_jpg")

            # The part we want to keep is the first item (index 0)
            clean_name_prefix = parts[0]
            extension = parts[1][-3:]
            
            # Construct the new full name with extension
            new_filename = f"{clean_name_prefix}.{extension}"

            # Create full file paths
            old_file_path = os.path.join(folder_path, filename)
            new_file_path = os.path.join(folder_path, new_filename)

            if dry_run:
                print(f"[DRY RUN] Would rename: '{filename}'  -->  '{new_filename}'")
            else:
                # Perform the rename
                try:
                    os.rename(old_file_path, new_file_path)
                    print(f"[SUCCESS] Renamed: '{filename}'  -->  '{new_filename}'")
                except FileExistsError:
                    print(f"[SKIPPED] '{new_filename}' already exists. Skipping to avoid overwrite.")
                except Exception as e:
                    print(f"[ERROR] Could not rename '{filename}': {e}")

            count += 1

    print("-" * 50)
    if count == 0:
        print("No files matching the pattern were found.")
    else:
        action = "would be" if dry_run else "were"
        print(f"Process complete. {count} files {action} renamed.")


if __name__ == "__main__":
    rename_images(sys.argv[1], bool(sys.argv[2] == "True"))