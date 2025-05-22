import os
import sys
import re
import subprocess
import pkg_resources

def find_imports(directory):
    """Find all imports in Python files in the given directory."""
    imports = set()
    
    # Common Python standard library modules to exclude
    std_lib = {
        'os', 'sys', 're', 'time', 'datetime', 'json', 'math', 'random', 
        'logging', 'traceback', 'pathlib', 'tempfile', 'shutil', 'gc', 
        'multiprocessing', 'concurrent', 'typing', 'collections', 'functools'
    }
    
    # Walk through all Python files in the directory
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                        # Find import statements
                        import_lines = re.findall(r'^import\s+([\w\.]+)', content, re.MULTILINE)
                        from_import_lines = re.findall(r'^from\s+([\w\.]+)\s+import', content, re.MULTILINE)
                        
                        # Add to set of imports
                        for imp in import_lines:
                            base_module = imp.split('.')[0]
                            if base_module not in std_lib:
                                imports.add(base_module)
                        
                        for imp in from_import_lines:
                            base_module = imp.split('.')[0]
                            if base_module not in std_lib:
                                imports.add(base_module)
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
    
    return imports

def get_installed_version(package):
    """Get the installed version of a package."""
    try:
        return pkg_resources.get_distribution(package).version
    except pkg_resources.DistributionNotFound:
        return None

def generate_requirements(directory, output_file='requirements.txt'):
    """Generate requirements.txt file based on imports found in the directory."""
    imports = find_imports(directory)
    
    # Add known dependencies that might not be directly imported
    known_dependencies = {
        'fastapi': '0.104.1',
        'uvicorn': '0.23.2',
        'python-multipart': '0.0.6',
        'pydantic': '2.4.2',
        'huggingface_hub': '0.16.4',  # Specific version for cached_download
        'datasets': '2.14.5'
    }
    
    # Combine found imports with known dependencies
    all_packages = set(imports)
    for dep in known_dependencies:
        all_packages.add(dep)
    
    # Get versions for all packages
    requirements = []
    for package in sorted(all_packages):
        if package in known_dependencies:
            version = known_dependencies[package]
            requirements.append(f"{package}=={version}")
        else:
            version = get_installed_version(package)
            if version:
                requirements.append(f"{package}=={version}")
            else:
                # If package is not installed but found in imports, add without version
                requirements.append(package)
    
    # Write to requirements.txt
    with open(output_file, 'w') as f:
        f.write('\n'.join(requirements))
    
    print(f"Generated {output_file} with {len(requirements)} packages")
    return requirements

if __name__ == "__main__":
    directory = "."  # Current directory
    if len(sys.argv) > 1:
        directory = sys.argv[1]
    
    output_file = "requirements.txt"
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    
    generate_requirements(directory, output_file)