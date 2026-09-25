import os
import argparse
import sys


def get_args():
    """
    Configures and parses command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Modify YOLO class indices in text files. Can set a single class or swap two classes."
    )

    # Required argument: The path (file or folder)
    parser.add_argument(
        "path",
        type=str,
        help="Path to a specific .txt file OR a folder containing .txt files."
    )

    parser.add_argument(
        "--recurse",
        action="store_true",
        help="When path is a folder, process .txt files in all subfolders too."
    )

    # Create a group so user can't choose BOTH set-class AND swap
    group = parser.add_mutually_exclusive_group(required=True)

    # Option A: Set all to one class
    group.add_argument(
        "--set-class",
        type=int,
        help="New class index. Will overwrite ALL objects in the file(s) to this index."
    )

    # Option B: Swap two classes
    group.add_argument(
        "--swap",
        type=int,
        nargs=2,
        metavar=('CLASS_A', 'CLASS_B'),
        help="Swap two class indices. Objects of Class A become B, and B become A."
    )

    # Option C: Change a class
    group.add_argument(
        "--change",
        type=int,
        nargs=2,
        metavar=('CLASS_A', 'CLASS_B'),
        help="Change two class indices. Objects of Class A become B"
    )

    return parser.parse_args()


def process_file(file_path, mode, **kwargs):
    """
    Reads a file, applies the logic, and writes it back.
    Returns True if file was modified, False otherwise.
    """
    modified = False
    new_lines = []

    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()

            # Validation: YOLO lines must have at least 5 parts (class x y w h)
            if len(parts) >= 5:
                current_id = int(parts[0])
                new_id = current_id

                # --- LOGIC: Set Single Class ---
                if mode == 'set':
                    target_id = kwargs.get('target_id')
                    if current_id != target_id:
                        new_id = target_id
                        modified = True

                # --- LOGIC: Swap Classes ---
                elif mode == 'swap':
                    swap_a = kwargs.get('swap_a')
                    swap_b = kwargs.get('swap_b')

                    if current_id == swap_a:
                        new_id = swap_b
                        modified = True
                    elif current_id == swap_b:
                        new_id = swap_a
                        modified = True
                elif mode == 'change':
                    source_id = kwargs.get('source_id')
                    target_id = kwargs.get('target_id')
                    if current_id == source_id:
                        new_id = target_id
                        modified = True

                # Update the line if changed
                parts[0] = str(new_id)
                new_line = " ".join(parts) + "\n"
                new_lines.append(new_line)
            else:
                # Keep malformed or empty lines as-is to prevent data loss
                new_lines.append(line)

        # Only write back if changes were actually made
        if modified:
            with open(file_path, 'w') as f:
                f.writelines(new_lines)
            return True

    except Exception as e:
        print(f"[ERROR] processing {os.path.basename(file_path)}: {e}")
        return False

    return False


def main():
    args = get_args()

    # Determine the list of files to process
    files_to_process = []

    if os.path.isfile(args.path):
        # User provided a single file
        files_to_process.append(args.path)
    elif os.path.isdir(args.path):
        # User provided a directory
        if args.recurse:
            for root, _, filenames in os.walk(args.path):
                for f in filenames:
                    if f.endswith(".txt") and f != "classes.txt":
                        files_to_process.append(os.path.join(root, f))
        else:
            for f in os.listdir(args.path):
                if f.endswith(".txt") and f != "classes.txt":
                    files_to_process.append(os.path.join(args.path, f))
    else:
        print(f"Error: The path '{args.path}' does not exist.")
        sys.exit(1)

    # Determine Mode
    mode = None
    params = {}

    if args.set_class is not None:
        mode = 'set'
        params['target_id'] = args.set_class
        print(f"Mode: SET ALL to class {args.set_class}")
    elif args.swap:
        mode = 'swap'
        params['swap_a'] = args.swap[0]
        params['swap_b'] = args.swap[1]
        print(f"Mode: SWAP class {args.swap[0]} <-> {args.swap[1]}")
    elif args.change:
        mode = 'change'
        params['source_id'] = args.change[0]
        params['target_id'] = args.change[1]
        print(f"Mode: CHANGE class {args.change[0]} -> {args.change[1]}")

    print(f"Target: {args.path}")
    print("-" * 40)

    # Run Loop
    count = 0
    for file_path in files_to_process:
        if process_file(file_path, mode, **params):
            print(f"[UPDATED] {os.path.basename(file_path)}")
            count += 1

    print("-" * 40)
    print(f"Processing complete. Modified {count} files.")


if __name__ == "__main__":
    main()
