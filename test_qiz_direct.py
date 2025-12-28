#!/usr/bin/env python3
"""
Direct test script for QIZ module multilingual support
"""

import sys
import os

# Add the current directory to the path so we can import aiplatform
current_dir = os.path.dirname(os.path.abspath(__file__))
aiplatform_dir = os.path.join(current_dir, 'aiplatform')

# Temporarily modify sys.path to avoid package imports
original_path = sys.path[:]
sys.path.insert(0, current_dir)

def test_qiz_direct():
    """Test QIZ module with direct imports"""
    try:
        # Import the QIZ module directly
        import importlib.util
        
        # Load the QIZ module directly
        qiz_path = os.path.join(aiplatform_dir, 'qiz.py')
        spec = importlib.util.spec_from_file_location("qiz", qiz_path)
        qiz_module = importlib.util.module_from_spec(spec)
        
        # Execute the module
        spec.loader.exec_module(qiz_module)
        
        print("✓ QIZ module loaded directly")
        
        # Test creating QMP node
        QMP = qiz_module.QuantumMeshProtocol
        qmp_en = QMP("test_node_en", language="en")
        print("✓ English QMP node created")
        
        qmp_ru = QMP("test_node_ru", language="ru")
        print("✓ Russian QMP node created")
        
        qmp_zh = QMP("test_node_zh", language="zh")
        print("✓ Chinese QMP node created")
        
        qmp_ar = QMP("test_node_ar", language="ar")
        print("✓ Arabic QMP node created")
        
        # Test quantum signature
        QuantumSignature = qiz_module.QuantumSignature
        sig_en = QuantumSignature("test_data_en", language="en")
        sig_ru = QuantumSignature("test_data_ru", language="ru")
        sig_zh = QuantumSignature("test_data_zh", language="zh")
        sig_ar = QuantumSignature("test_data_ar", language="ar")
        
        print("✓ Quantum signatures created with multilingual support")
        
        # Test adding neighbors
        qmp_en.add_neighbor("neighbor_en", "192.168.1.1", sig_en)
        qmp_ru.add_neighbor("neighbor_ru", "192.168.1.2", sig_ru)
        qmp_zh.add_neighbor("neighbor_zh", "192.168.1.3", sig_zh)
        qmp_ar.add_neighbor("neighbor_ar", "192.168.1.4", sig_ar)
        
        print("✓ Neighbors added with multilingual support")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed to test QIZ module directly: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Restore original sys.path
        sys.path[:] = original_path

if __name__ == "__main__":
    print("Testing QIZ module multilingual support (direct import)...")
    
    if test_qiz_direct():
        print("\n✓ QIZ multilingual support test passed!")
    else:
        print("\n✗ QIZ multilingual support test failed!")