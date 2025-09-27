import os
import sys
import warnings

def setup_libs():
    """
    Hybrid approach: Use QGIS libraries where possible, but allow full PyTorch ecosystem.
    This manages the NumPy compatibility issue between QGIS and PyTorch/torchvision.
    """
    # Find libs directory
    current_dir = os.path.dirname(__file__)
    libs_dir = None
    
    while current_dir != os.path.dirname(current_dir):
        potential_libs = os.path.join(current_dir, 'libs')
        if os.path.exists(potential_libs):
            libs_dir = potential_libs
            break
        current_dir = os.path.dirname(current_dir)
    
    if not libs_dir:
        return
    
    # CRITICAL: Handle NumPy compatibility
    _setup_numpy_compatibility(libs_dir)
    
    # Add libs to path
    if libs_dir not in sys.path:
        sys.path.insert(0, libs_dir)

def _setup_numpy_compatibility(libs_dir):
    """
    Handle NumPy version conflicts between QGIS and PyTorch/torchvision.
    """
    # Check if we have NumPy in libs (from PyTorch dependencies)
    plugin_numpy_path = os.path.join(libs_dir, 'numpy')
    
    if os.path.exists(plugin_numpy_path):
        # We have NumPy from PyTorch - need to manage compatibility
        
        # Try to use QGIS NumPy first
        try:
            # Import QGIS NumPy without our libs in path
            temp_path = sys.path.copy()
            filtered_path = [p for p in sys.path if 'libs' not in p]
            sys.path = filtered_path
            
            import numpy as qgis_numpy
            qgis_version = qgis_numpy.__version__
            
            # Restore path
            sys.path = temp_path
            
            # Now check plugin NumPy version
            sys.path.insert(0, libs_dir)
            import numpy as plugin_numpy
            plugin_version = plugin_numpy.__version__
            
            print(f"QGIS NumPy: {qgis_version}, Plugin NumPy: {plugin_version}")
            
            # If versions are close enough, prefer QGIS version
            if _numpy_versions_compatible(qgis_version, plugin_version):
                # Remove plugin numpy from path to use QGIS version
                if plugin_numpy_path in sys.path:
                    sys.path.remove(plugin_numpy_path)
                print("Using QGIS NumPy (compatible versions)")
            else:
                print("Using plugin NumPy (version incompatible)")
                
        except Exception as e:
            print(f"NumPy compatibility check failed: {e}")
            # Fall back to plugin version
            pass

def _numpy_versions_compatible(qgis_version, plugin_version):
    """Check if NumPy versions are compatible enough"""
    try:
        def version_tuple(v):
            return tuple(map(int, v.split('.')[:2]))  # Compare major.minor only
        
        qgis_ver = version_tuple(qgis_version)
        plugin_ver = version_tuple(plugin_version)
        
        # Compatible if major version same and minor within 2
        if qgis_ver[0] == plugin_ver[0]:  # Same major version
            return abs(qgis_ver[1] - plugin_ver[1]) <= 2  # Minor within 2
        return False
    except:
        return False

def safe_import_ml_libraries():
    """
    Get ML libraries using hybrid approach - QGIS where possible, plugin libs for PyTorch ecosystem.
    """
    libraries = {}
    
    # Always use QGIS versions for these (avoid conflicts)
    qgis_preferred = ['scipy', 'cv2', 'shapely', 'fiona', 'rasterio']
    
    for lib_name in qgis_preferred:
        libraries[lib_name] = _get_qgis_library(lib_name)
    
    # Handle NumPy specially (may use either QGIS or plugin version)
    libraries['numpy'] = _get_compatible_numpy()
    
    # Use plugin versions for PyTorch ecosystem
    pytorch_libs = ['torch', 'torchvision', 'torchmetrics', 'wandb', 'yacs', 'tqdm']
    
    for lib_name in pytorch_libs:
        try:
            lib = __import__(lib_name)
            libraries[lib_name] = lib
            if lib_name == 'torch':
                print(f"PyTorch {lib.__version__} - CUDA: {lib.cuda.is_available()}")
        except ImportError:
            libraries[lib_name] = None
            print(f"{lib_name} not available")
    
    # Handle Pillow specially - prefer QGIS but allow fallback
    libraries['PIL'] = _get_compatible_pillow()
    
    return libraries

def _get_qgis_library(lib_name):
    """Get library from QGIS, excluding plugin libs from path"""
    try:
        temp_path = sys.path.copy()
        filtered_path = [p for p in sys.path if 'libs' not in p]
        sys.path = filtered_path
        
        if lib_name == 'cv2':
            import cv2
            lib = cv2
        elif lib_name == 'matplotlib':
            import matplotlib
            lib = matplotlib
        else:
            lib = __import__(lib_name)
        
        sys.path = temp_path
        print(f"Using QGIS {lib_name}")
        return lib
        
    except ImportError:
        sys.path = temp_path
        try:
            lib = __import__(lib_name)
            print(f"Using plugin {lib_name}")
            return lib
        except ImportError:
            print(f"{lib_name} not available")
            return None

def _get_compatible_numpy():
    """Get NumPy version that works with both QGIS and PyTorch"""
    try:
        import numpy
        print(f"Using NumPy {numpy.__version__}")
        return numpy
    except ImportError:
        print("NumPy not available")
        return None

def _get_compatible_pillow():
    """Get Pillow version that works with both QGIS and torchvision"""
    # Try QGIS version first
    try:
        temp_path = sys.path.copy()
        filtered_path = [p for p in sys.path if 'libs' not in p]
        sys.path = filtered_path
        
        import PIL
        sys.path = temp_path
        print("Using QGIS Pillow")
        return PIL
        
    except ImportError:
        sys.path = temp_path
        try:
            import PIL
            print("Using plugin Pillow")
            return PIL
        except ImportError:
            print("Pillow not available")
            return None

# Backwards compatibility - keep the old automatic call commented out
# setup_libs()