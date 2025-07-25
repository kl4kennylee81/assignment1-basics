#!/usr/bin/env python3
"""
Script to fix Jupyter notebook widget rendering issues on GitHub
by cleaning widget metadata that causes rendering problems.
"""

import json
import sys
from pathlib import Path

def clean_notebook_widgets(notebook_path):
    """
    Clean widget metadata from a Jupyter notebook to fix GitHub rendering issues.
    
    Args:
        notebook_path: Path to the .ipynb file
    """
    try:
        # Read the notebook
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
        
        print(f"Processing: {notebook_path}")
        
        # Clean cell metadata
        cells_cleaned = 0
        for cell in notebook.get('cells', []):
            if 'metadata' in cell:
                # Remove widget-related metadata that causes issues
                metadata = cell['metadata']
                
                # Remove widget metadata
                if 'widgets' in metadata:
                    del metadata['widgets']
                    cells_cleaned += 1
                
                # Clean up execution metadata that can cause widget issues
                if 'executionInfo' in metadata:
                    del metadata['executionInfo']
                
                # Remove empty metadata
                if not metadata:
                    del cell['metadata']
        
        # Clean notebook-level widget metadata
        notebook_cleaned = False
        if 'metadata' in notebook:
            nb_metadata = notebook['metadata']
            
            # Remove widget state that causes GitHub rendering issues
            if 'widgets' in nb_metadata:
                del nb_metadata['widgets']
                notebook_cleaned = True
            
            # Clean up kernel metadata that can interfere
            if 'kernelspec' in nb_metadata:
                # Keep kernelspec but clean problematic fields
                kernelspec = nb_metadata['kernelspec']
                # Remove any widget-related kernel metadata
                if 'widgets' in kernelspec:
                    del kernelspec['widgets']
        
        # Clean output widget data
        outputs_cleaned = 0
        for cell in notebook.get('cells', []):
            if cell.get('cell_type') == 'code' and 'outputs' in cell:
                for output in cell['outputs']:
                    # Remove widget view outputs that cause rendering issues
                    if output.get('output_type') == 'display_data':
                        if 'data' in output:
                            data = output['data']
                            # Remove widget view data
                            widget_keys = [k for k in data.keys() if 'widget' in k.lower()]
                            for key in widget_keys:
                                del data[key]
                                outputs_cleaned += 1
                    
                    # Clean widget metadata from outputs
                    if 'metadata' in output:
                        out_metadata = output['metadata']
                        if 'widgets' in out_metadata:
                            del out_metadata['widgets']
                            outputs_cleaned += 1
        
        # Write the cleaned notebook
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=1, ensure_ascii=False)
        
        print(f"✓ Cleaned {cells_cleaned} cells, {outputs_cleaned} outputs")
        if notebook_cleaned:
            print("✓ Cleaned notebook-level widget metadata")
        print(f"✓ Fixed: {notebook_path}")
        
    except json.JSONDecodeError as e:
        print(f"✗ Error: Invalid JSON in {notebook_path}: {e}")
        return False
    except Exception as e:
        print(f"✗ Error processing {notebook_path}: {e}")
        return False
    
    return True

def main():
    if len(sys.argv) != 2:
        print("Usage: python fix_notebook_widgets.py <notebook.ipynb>")
        print("   or: python fix_notebook_widgets.py <directory>")
        sys.exit(1)
    
    path = Path(sys.argv[1])
    
    if path.is_file() and path.suffix == '.ipynb':
        # Single notebook file
        success = clean_notebook_widgets(path)
        sys.exit(0 if success else 1)
    
    elif path.is_dir():
        # Directory - process all notebooks
        notebook_files = list(path.glob('**/*.ipynb'))
        if not notebook_files:
            print(f"No .ipynb files found in {path}")
            sys.exit(1)
        
        print(f"Found {len(notebook_files)} notebook(s)")
        
        success_count = 0
        for notebook_file in notebook_files:
            if clean_notebook_widgets(notebook_file):
                success_count += 1
            print()  # Empty line between files
        
        print(f"Successfully processed {success_count}/{len(notebook_files)} notebooks")
        sys.exit(0 if success_count == len(notebook_files) else 1)
    
    else:
        print(f"Error: {path} is not a valid .ipynb file or directory")
        sys.exit(1)

if __name__ == "__main__":
    main()