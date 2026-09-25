import os
import shutil


def update_existing_files():
    """
    Updates files in the destination directory if they exist, using the corresponding files from the source directory.
    Files not present in the destination are ignored, with no action taken. This operation also preserves file metadata
    during the update process.

    :param None

    :raises FileNotFoundError: Raised when the provided source or destination path does not exist.

    :return: None
    """
    print("--- Update Existing Files Only ---")

    # 1. Get paths from the user
    # We use .strip() to remove accidental spaces at the start/end
    source_dir = input("Enter the SOURCE path: ").strip()
    dest_dir = input("Enter the DESTINATION path: ").strip()

    # 2. Validate paths
    if not os.path.exists(source_dir):
        print(f"Error: Source path does not exist: {source_dir}")
        return
    if not os.path.exists(dest_dir):
        print(f"Error: Destination path does not exist: {dest_dir}")
        return

    print(f"\nScanning source: {source_dir}...")

    files_updated = 0
    files_skipped = 0

    # 3. Iterate through files in the source directory
    for filename in os.listdir(source_dir):

        # Construct full file paths
        source_file = os.path.join(source_dir, filename)
        dest_file = os.path.join(dest_dir, filename)

        # Ensure we are dealing with a file, not a subfolder
        if os.path.isfile(source_file):

            # 4. CHECK: Does this file exist in the destination?
            if os.path.exists(dest_file):
                try:
                    # Copy the file and preserve metadata (timestamps)
                    shutil.copy2(source_file, dest_file)
                    print(f"[UPDATED] {filename}")
                    files_updated += 1
                except Exception as e:
                    print(f"[ERROR] Could not update {filename}: {e}")
            else:
                # File does not exist in destination, so we do nothing
                files_skipped += 1

    # 5. Summary
    print("-" * 30)
    print(f"Process complete.")
    print(f"Files overwritten/updated: {files_updated}")
    print(f"Files skipped (not in destination): {files_skipped}")


if __name__ == "__main__":
    update_existing_files()