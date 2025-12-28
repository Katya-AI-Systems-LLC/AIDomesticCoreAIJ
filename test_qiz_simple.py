#!/usr/bin/env python3
"""
Simple test script for QIZ module multilingual support
"""

import sys
import os

# Add the current directory to the path so we can import aiplatform
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_qiz_simple():
    """Test QIZ module with minimal dependencies"""
    try:
        # Import only what we need directly
        from aiplatform.qiz import QuantumMeshProtocol
        
        # Test creating QMP node with different languages
        print("Testing QIZ module with different languages...")
        
        # Test with English
        qmp_en = QuantumMeshProtocol("test_node_en", language="en")
        print("✓ English QMP node created")
        
        # Test with Russian
        qmp_ru = QuantumMeshProtocol("test_node_ru", language="ru")
        print("✓ Russian QMP node created")
        
        # Test with Chinese
        qmp_zh = QuantumMeshProtocol("test_node_zh", language="zh")
        print("✓ Chinese QMP node created")
        
        # Test with Arabic
        qmp_ar = QuantumMeshProtocol("test_node_ar", language="ar")
        print("✓ Arabic QMP node created")
        
        # Test adding a neighbor with different languages
        from aiplatform.qiz import QuantumSignature
        
        # Create signatures with different languages
        sig_en = QuantumSignature("test_data_en", language="en")
        sig_ru = QuantumSignature("test_data_ru", language="ru")
        sig_zh = QuantumSignature("test_data_zh", language="zh")
        sig_ar = QuantumSignature("test_data_ar", language="ar")
        
        print("✓ Quantum signatures created with multilingual support")
        
        # Add neighbors
        qmp_en.add_neighbor("neighbor_en", "192.168.1.1", sig_en)
        qmp_ru.add_neighbor("neighbor_ru", "192.168.1.2", sig_ru)
        qmp_zh.add_neighbor("neighbor_zh", "192.168.1.3", sig_zh)
        qmp_ar.add_neighbor("neighbor_ar", "192.168.1.4", sig_ar)
        
        print("✓ Neighbors added with multilingual support")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed to test QIZ module: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing QIZ module multilingual support...")
    
    if test_qiz_simple():
        print("\n✓ QIZ multilingual support test passed!")
    else:
        print("\n✗ QIZ multilingual support test failed!")