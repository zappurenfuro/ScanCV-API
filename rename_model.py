import os
import shutil

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Source file (the uploaded model)
source_file = os.path.join(current_dir, "scp.py")

# Destination file (the name we're importing in the FastAPI app)
dest_file = os.path.join(current_dir, "resume_scanner.py")

# Rename the file if it exists
if os.path.exists(source_file):
    shutil.copy(source_file, dest_file)
    print(f"Renamed {source_file} to {dest_file}")
else:
    print(f"Source file {source_file} not found")
