#!/usr/bin/env python3
"""
Test script for QIZ module multilingual support
"""

import sys
import os

# Add the current directory to the path so we can import aiplatform
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_qiz_import():
    """Test importing QIZ module directly"""
    try:
        # Import the QIZ module directly without going through __init__.py
        from aiplatform.qiz import QuantumMeshProtocol
        print("✓ QIZ module imported successfully")
        return True
    except Exception as e:
        print(f"✗ Failed to import QIZ module: {e}")
        return False

def test_qiz_multilingual():
    """Test QIZ module multilingual support"""
    try:
        from aiplatform.qiz import QuantumMeshProtocol
        
        # Test with English
        qmp_en = QuantumMeshProtocol("test_node", language="en")
        print("✓ QIZ module created with English language support")
        
        # Test with Russian
        qmp_ru = QuantumMeshProtocol("test_node", language="ru")
        print("✓ QIZ module created with Russian language support")
        
        # Test with Chinese
        qmp_zh = QuantumMeshProtocol("test_node", language="zh")
        print("✓ QIZ module created with Chinese language support")
        
        # Test with Arabic
        qmp_ar = QuantumMeshProtocol("test_node", language="ar")
        print("✓ QIZ module created with Arabic language support")
        
        return True
    except Exception as e:
        print(f"✗ Failed to test QIZ multilingual support: {e}")
        return False

if __name__ == "__main__":
    print("Testing QIZ module...")
    
    if test_qiz_import():
        print("QIZ import test passed")
    
    if test_qiz_multilingual():
        print("QIZ multilingual test passed")
    
    print("QIZ testing completed")