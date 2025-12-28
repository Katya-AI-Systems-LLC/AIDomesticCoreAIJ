"""
Quantum Infrastructure Zero (QIZ) Example for AIPlatform SDK

This example demonstrates QIZ capabilities with multilingual support.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aiplatform.qiz import (
    create_zero_dns,
    create_qmp_node,
    create_zero_server,
    create_post_dns_logic,
    create_deploy_engine,
    create_zero_trust_model
)
import json


def zero_dns_example(language='en'):
    """Demonstrate Zero-DNS functionality."""
    print(f"=== {translate('zero_dns_example', language) or 'Zero-DNS Example'} ===")
    
    # Create Zero-DNS system
    dns = create_zero_dns(language=language)
    
    # Register some entries
    service_data = {"host": "192.168.1.100", "port": 8080}
    signature1 = dns.register("web_service", service_data, {"type": "web_server"})
    
    database_data = {"host": "192.168.1.101", "port": 5432}
    signature2 = dns.register("database", database_data, {"type": "postgresql"})
    
    print(f"Registered web_service with signature: {signature1}")
    print(f"Registered database with signature: {signature2}")
    
    # Resolve entries
    web_entry = dns.resolve("web_service")
    db_entry = dns.resolve("database")
    
    if web_entry:
        print(f"Resolved web_service: {web_entry.to_dict()}")
    
    if db_entry:
        print(f"Resolved database: {db_entry.to_dict()}")
    
    # List entries
    entries = dns.list_entries()
    print(f"DNS entries: {entries}")
    print()


def qmp_example(language='en'):
    """Demonstrate Quantum Mesh Protocol."""
    print(f"=== {translate('qmp_example', language) or 'Quantum Mesh Protocol Example'} ===")
    
    # Create QMP nodes
    node1 = create_qmp_node("node_1", language=language)
    node2 = create_qmp_node("node_2", language=language)
    
    # Add neighbors
    from aiplatform.qiz import QuantumSignature
    sig1 = QuantumSignature("node_1_data", language=language)
    sig2 = QuantumSignature("node_2_data", language=language)
    
    node1.add_neighbor("node_2", "192.168.1.2", sig2)
    node2.add_neighbor("node_1", "192.168.1.1", sig1)
    
    # Update routing tables
    node1.update_routing_table("node_2", "node_2", 1.0)
    node2.update_routing_table("node_1", "node_1", 1.0)
    
    # Route a message
    message = {"data": "Hello from node 1", "timestamp": "2025-01-01T00:00:00Z"}
    route_result = node1.route_message("node_2", message)
    print(f"Route result: {route_result}")
    print()


def zero_server_example(language='en'):
    """Demonstrate Zero-Server architecture."""
    print(f"=== {translate('zero_server_example', language) or 'Zero-Server Example'} ===")
    
    # Create Zero-Server
    server = create_zero_server("my_server", language=language)
    
    # Register services
    web_service = {"type": "flask", "port": 8080}
    ml_service = {"type": "tensorflow", "model": "resnet50"}
    
    sig1 = server.register_service("web_app", web_service, {"version": "1.0"})
    sig2 = server.register_service("ml_model", ml_service, {"version": "2.0"})
    
    print(f"Registered web_app with signature: {sig1}")
    print(f"Registered ml_model with signature: {sig2}")
    
    # Get services
    web_app = server.get_service("web_app")
    ml_model = server.get_service("ml_model")
    
    print(f"Web app: {web_app}")
    print(f"ML model: {ml_model}")
    
    # List services
    services = server.list_services()
    print(f"Services: {services}")
    print()


def post_dns_logic_example(language='en'):
    """Demonstrate Post-DNS logic layer."""
    print(f"=== {translate('post_dns_logic_example', language) or 'Post-DNS Logic Example'} ===")
    
    # Create Post-DNS logic layer
    logic = create_post_dns_logic(language=language)
    
    # Add a simple rule
    def condition(context):
        return context.get('request_type') == 'high_priority'
    
    def action(context):
        return {"route": "priority_queue", "priority": 1}
    
    logic.add_rule("priority_routing", condition, action)
    
    # Evaluate rules
    context1 = {"request_type": "high_priority", "data": "important_data"}
    context2 = {"request_type": "normal", "data": "regular_data"}
    
    results1 = logic.evaluate(context1)
    results2 = logic.evaluate(context2)
    
    print(f"High priority context results: {results1}")
    print(f"Normal context results: {results2}")
    print()


def deploy_engine_example(language='en'):
    """Demonstrate Self-Contained Deploy Engine."""
    print(f"=== {translate('deploy_engine_example', language) or 'Deploy Engine Example'} ===")
    
    # Create deploy engine
    engine = create_deploy_engine(language=language)
    
    # Deploy applications
    web_config = {
        "image": "nginx:latest",
        "ports": [80],
        "environment": {"ENV": "production"}
    }
    
    ml_config = {
        "image": "tensorflow/serving:latest",
        "ports": [8501],
        "model_path": "/models"
    }
    
    web_deployment = engine.deploy("web_app", web_config)
    ml_deployment = engine.deploy("ml_service", ml_config)
    
    print(f"Web deployment: {web_deployment}")
    print(f"ML deployment: {ml_deployment}")
    
    # Note: In a real implementation, undeploy would be called when needed
    print()


def zero_trust_example(language='en'):
    """Demonstrate Zero-Trust security model."""
    print(f"=== {translate('zero_trust_example', language) or 'Zero-Trust Example'} ===")
    
    # Create Zero-Trust model
    trust_model = create_zero_trust_model(language=language)
    
    # Add policies
    web_policy = {
        "subject": "web_user",
        "resource": "web_app",
        "action": "read",
        "allow": True
    }
    
    db_policy = {
        "subject": "admin",
        "resource": "database",
        "action": "*",
        "allow": True
    }
    
    trust_model.add_policy("web_access", web_policy)
    trust_model.add_policy("db_admin", db_policy)
    
    # Validate access
    web_access = trust_model.validate_access("web_user", "web_app", "read")
    db_access = trust_model.validate_access("web_user", "database", "write")
    admin_db_access = trust_model.validate_access("admin", "database", "write")
    
    print(f"Web user read web_app: {web_access}")
    print(f"Web user write database: {db_access}")
    print(f"Admin write database: {admin_db_access}")
    print()


def translate(key, language):
    """Simple translation function for example titles."""
    translations = {
        'zero_dns_example': {
            'ru': 'Пример Zero-DNS',
            'zh': 'Zero-DNS示例',
            'ar': 'مثال Zero-DNS'
        },
        'qmp_example': {
            'ru': 'Пример Quantum Mesh Protocol',
            'zh': '量子网格协议示例',
            'ar': 'مثال بروتوكول الشبكة الكمومية'
        },
        'zero_server_example': {
            'ru': 'Пример Zero-Server',
            'zh': 'Zero-Server示例',
            'ar': 'مثال Zero-Server'
        },
        'post_dns_logic_example': {
            'ru': 'Пример Post-DNS логики',
            'zh': 'Post-DNS逻辑示例',
            'ar': 'مثال منطق Post-DNS'
        },
        'deploy_engine_example': {
            'ru': 'Пример движка развертывания',
            'zh': '部署引擎示例',
            'ar': 'مثال محرك النشر'
        },
        'zero_trust_example': {
            'ru': 'Пример модели Zero-Trust',
            'zh': 'Zero-Trust模型示例',
            'ar': 'مثال نموذج Zero-Trust'
        }
    }
    
    if key in translations and language in translations[key]:
        return translations[key][language]
    return None


def main():
    """Run all QIZ examples."""
    languages = ['en', 'ru', 'zh', 'ar']
    
    for language in languages:
        print(f"\n{'='*50}")
        print(f"QIZ EXAMPLES - {language.upper()}")
        print(f"{'='*50}\n")
        
        try:
            zero_dns_example(language)
            qmp_example(language)
            zero_server_example(language)
            post_dns_logic_example(language)
            deploy_engine_example(language)
            zero_trust_example(language)
        except Exception as e:
            print(f"Error in {language} examples: {e}")


if __name__ == "__main__":
    main()