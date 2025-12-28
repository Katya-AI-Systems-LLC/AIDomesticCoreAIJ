"""
Vocabulary Management System for AIPlatform SDK

This module provides technical vocabulary management capabilities for the AIPlatform Quantum Infrastructure Zero SDK,
supporting specialized terminology for Russian, Chinese, and Arabic languages in quantum computing and AI domains.
"""

from typing import Dict, List, Set, Optional
from collections import defaultdict
import re


class VocabularyManager:
    """Technical vocabulary management system for multilingual support."""
    
    def __init__(self):
        """Initialize the vocabulary manager."""
        self.supported_languages = ['en', 'ru', 'zh', 'ar']
        self.domains = ['quantum', 'ai', 'security', 'networking', 'computing', 'qiz']
        
        # Load technical vocabularies
        self.vocabularies = defaultdict(dict)  # domain -> language -> terms
        self._initialize_vocabularies()
    
    def _initialize_vocabularies(self):
        """Initialize technical vocabularies for all domains and languages."""
        # Quantum computing vocabulary
        self._initialize_quantum_vocabulary()
        
        # AI/ML vocabulary
        self._initialize_ai_vocabulary()
        
        # Security vocabulary
        self._initialize_security_vocabulary()
        
        # Networking vocabulary
        self._initialize_networking_vocabulary()
        
        # General computing vocabulary
        self._initialize_computing_vocabulary()
        
        # QIZ vocabulary
        self._initialize_qiz_vocabulary()
    
    def _initialize_quantum_vocabulary(self):
        """Initialize quantum computing vocabulary."""
        quantum_terms = {
            'en': {
                'quantum_computing': 'Quantum Computing',
                'quantum_bit': 'Qubit',
                'quantum_circuit': 'Quantum Circuit',
                'quantum_gate': 'Quantum Gate',
                'superposition': 'Superposition',
                'entanglement': 'Entanglement',
                'quantum_algorithm': 'Quantum Algorithm',
                'quantum_processor': 'Quantum Processor',
                'quantum_state': 'Quantum State',
                'quantum_measurement': 'Quantum Measurement',
                'hadamard_gate': 'Hadamard Gate',
                'pauli_x_gate': 'Pauli-X Gate',
                'pauli_y_gate': 'Pauli-Y Gate',
                'pauli_z_gate': 'Pauli-Z Gate',
                'cnot_gate': 'CNOT Gate',
                'quantum_simulator': 'Quantum Simulator',
                'variational_quantum_eigensolver': 'Variational Quantum Eigensolver',
                'quantum_approximate_optimization_algorithm': 'Quantum Approximate Optimization Algorithm',
                'quantum_fourier_transform': 'Quantum Fourier Transform',
                'quantum_phase_estimation': 'Quantum Phase Estimation',
                'quantum_error_correction': 'Quantum Error Correction',
                'quantum_annealing': 'Quantum Annealing',
                'quantum_supremacy': 'Quantum Supremacy',
                'noisy_intermediate_scale_quantum': 'Noisy Intermediate Scale Quantum',
                'quantum_volume': 'Quantum Volume',
                'quantum_advantage': 'Quantum Advantage',
                'variational_quantum_eigensolver': 'Variational Quantum Eigensolver',
                'quantum_approximate_optimization_algorithm': 'Quantum Approximate Optimization Algorithm',
                "grover's_search_algorithm": "Grover's Search Algorithm",
                "shor's_factoring_algorithm": "Shor's Factoring Algorithm",
                'quantum_safe_cryptography': 'Quantum-Safe Cryptography'
            },
            'ru': {
                'quantum_computing': 'Квантовые вычисления',
                'quantum_bit': 'Кубит',
                'quantum_circuit': 'Квантовая цепь',
                'quantum_gate': 'Квантовый гейт',
                'superposition': 'Суперпозиция',
                'entanglement': 'Запутанность',
                'quantum_algorithm': 'Квантовый алгоритм',
                'quantum_processor': 'Квантовый процессор',
                'quantum_state': 'Квантовое состояние',
                'quantum_measurement': 'Квантовое измерение',
                'hadamard_gate': 'Гейт Адамара',
                'pauli_x_gate': 'Гейт Паули-X',
                'pauli_y_gate': 'Гейт Паули-Y',
                'pauli_z_gate': 'Гейт Паули-Z',
                'cnot_gate': 'Гейт CNOT',
                'quantum_simulator': 'Квантовый симулятор',
                'variational_quantum_eigensolver': 'Вариационный квантовый собственный решатель',
                'quantum_approximate_optimization_algorithm': 'Квантовый приближенный алгоритм оптимизации',
                'quantum_fourier_transform': 'Квантовое преобразование Фурье',
                'quantum_phase_estimation': 'Оценка квантовой фазы',
                'quantum_error_correction': 'Квантовая коррекция ошибок',
                'quantum_annealing': 'Квантовый отжиг',
                'quantum_supremacy': 'Квантовое превосходство',
                'noisy_intermediate_scale_quantum': 'Шумные промежуточные масштабные квантовые',
                'quantum_volume': 'Квантовый объем',
                'quantum_advantage': 'Квантовое преимущество',
                'variational_quantum_eigensolver': 'Вариационный квантовый собственный решатель',
                'quantum_approximate_optimization_algorithm': 'Квантовый приближенный алгоритм оптимизации',
                "grover's_search_algorithm": 'Алгоритм поиска Гровера',
                "shor's_factoring_algorithm": 'Алгоритм факторизации Шора',
                'quantum_safe_cryptography': 'Квантово-безопасная криптография'
            },
            'zh': {
                'quantum_computing': '量子计算',
                'quantum_bit': '量子比特',
                'quantum_circuit': '量子电路',
                'quantum_gate': '量子门',
                'superposition': '叠加态',
                'entanglement': '纠缠',
                'quantum_algorithm': '量子算法',
                'quantum_processor': '量子处理器',
                'quantum_state': '量子态',
                'quantum_measurement': '量子测量',
                'hadamard_gate': '哈达玛门',
                'pauli_x_gate': '泡利X门',
                'pauli_y_gate': '泡利Y门',
                'pauli_z_gate': '泡利Z门',
                'cnot_gate': '受控非门',
                'quantum_simulator': '量子模拟器',
                'variational_quantum_eigensolver': '变分量子本征求解器',
                'quantum_approximate_optimization_algorithm': '量子近似优化算法',
                'quantum_fourier_transform': '量子傅里叶变换',
                'quantum_phase_estimation': '量子相位估计算法',
                'quantum_error_correction': '量子纠错',
                'quantum_annealing': '量子退火',
                'quantum_supremacy': '量子霸权',
                'noisy_intermediate_scale_quantum': '含噪声中等规模量子',
                'quantum_volume': '量子体积',
                'quantum_advantage': '量子优势',
                'variational_quantum_eigensolver': '变分量子本征求解器',
                'quantum_approximate_optimization_algorithm': '量子近似优化算法',
                "grover's_search_algorithm": 'Grover搜索算法',
                "shor's_factoring_algorithm": 'Shor因数分解算法',
                'quantum_safe_cryptography': '抗量子密码学'
            },
            'ar': {
                'quantum_computing': 'الحوسبة الكمية',
                'quantum_bit': 'البت الكمي',
                'quantum_circuit': 'الدائرة الكمية',
                'quantum_gate': 'البوابة الكمية',
                'superposition': 'التراكب',
                'entanglement': 'التشابك',
                'quantum_algorithm': 'الخوارزمية الكمية',
                'quantum_processor': 'المعالج الكمي',
                'quantum_state': 'الحالة الكمية',
                'quantum_measurement': 'القياس الكمي',
                'hadamard_gate': 'بوابة هادامارد',
                'pauli_x_gate': 'بوابة باولي-X',
                'pauli_y_gate': 'بوابة باولي-Y',
                'pauli_z_gate': 'بوابة باولي-Z',
                'cnot_gate': 'بوابة CNOT',
                'quantum_simulator': 'محاكي الكم',
                'variational_quantum_eigensolver': 'المحلل الذاتي الكمي Variational',
                'quantum_approximate_optimization_algorithm': 'خوارزمية التحسين الكمي التقريبي',
                'quantum_fourier_transform': 'تحويل فورييه الكمي',
                'quantum_phase_estimation': 'تقدير الطور الكمي',
                'quantum_error_correction': 'تصحيح الأخطاء الكمية',
                'quantum_annealing': 'ال退火 الكمي',
                'quantum_supremacy': 'الهيمنة الكمية',
                'noisy_intermediate_scale_quantum': 'الكمي المتوسط الحجم مع الضوضاء',
                'quantum_volume': 'الحجم الكمي',
                'quantum_advantage': 'الميزة الكمية',
                'variational_quantum_eigensolver': 'المحلل الذاتي الكمي Variational',
                'quantum_approximate_optimization_algorithm': 'خوارزمية التحسين الكمي التقريبي',
                "grover's_search_algorithm": 'خوارزمية بحث جروفر',
                "shor's_factoring_algorithm": 'خوارزمية تحليل عوامل شور',
                'quantum_safe_cryptography': 'التشفير الكمي الآمن'
            }
        }
        
        self.vocabularies['quantum'] = quantum_terms
    
    def _initialize_ai_vocabulary(self):
        """Initialize AI/ML vocabulary."""
        ai_terms = {
            'en': {
                'artificial_intelligence': 'Artificial Intelligence',
                'machine_learning': 'Machine Learning',
                'deep_learning': 'Deep Learning',
                'neural_network': 'Neural Network',
                'convolutional_neural_network': 'Convolutional Neural Network',
                'recurrent_neural_network': 'Recurrent Neural Network',
                'transformer': 'Transformer',
                'attention_mechanism': 'Attention Mechanism',
                'natural_language_processing': 'Natural Language Processing',
                'computer_vision': 'Computer Vision',
                'reinforcement_learning': 'Reinforcement Learning',
                'supervised_learning': 'Supervised Learning',
                'unsupervised_learning': 'Unsupervised Learning',
                'semi_supervised_learning': 'Semi-Supervised Learning',
                'federated_learning': 'Federated Learning',
                'generative_adversarial_network': 'Generative Adversarial Network',
                'variational_autoencoder': 'Variational Autoencoder',
                'support_vector_machine': 'Support Vector Machine',
                'random_forest': 'Random Forest',
                'gradient_boosting': 'Gradient Boosting',
                'ensemble_learning': 'Ensemble Learning',
                'feature_engineering': 'Feature Engineering',
                'hyperparameter_tuning': 'Hyperparameter Tuning',
                'cross_validation': 'Cross-Validation',
                'confusion_matrix': 'Confusion Matrix'
            },
            'ru': {
                'artificial_intelligence': 'Искусственный интеллект',
                'machine_learning': 'Машинное обучение',
                'deep_learning': 'Глубокое обучение',
                'neural_network': 'Нейронная сеть',
                'convolutional_neural_network': 'Сверточная нейронная сеть',
                'recurrent_neural_network': 'Рекуррентная нейронная сеть',
                'transformer': 'Трансформер',
                'attention_mechanism': 'Механизм внимания',
                'natural_language_processing': 'Обработка естественного языка',
                'computer_vision': 'Компьютерное зрение',
                'reinforcement_learning': 'Обучение с подкреплением',
                'supervised_learning': 'Обучение с учителем',
                'unsupervised_learning': 'Обучение без учителя',
                'semi_supervised_learning': 'Полуобучение',
                'federated_learning': 'Федеративное обучение',
                'generative_adversarial_network': 'Генеративная состязательная сеть',
                'variational_autoencoder': 'Вариационный автоэнкодер',
                'support_vector_machine': 'Метод опорных векторов',
                'random_forest': 'Случайный лес',
                'gradient_boosting': 'Градиентный бустинг',
                'ensemble_learning': 'Ансамблевое обучение',
                'feature_engineering': 'Инжиниринг признаков',
                'hyperparameter_tuning': 'Настройка гиперпараметров',
                'cross_validation': 'Перекрестная проверка',
                'confusion_matrix': 'Матрица ошибок'
            },
            'zh': {
                'artificial_intelligence': '人工智能',
                'machine_learning': '机器学习',
                'deep_learning': '深度学习',
                'neural_network': '神经网络',
                'convolutional_neural_network': '卷积神经网络',
                'recurrent_neural_network': '循环神经网络',
                'transformer': '变压器',
                'attention_mechanism': '注意力机制',
                'natural_language_processing': '自然语言处理',
                'computer_vision': '计算机视觉',
                'reinforcement_learning': '强化学习',
                'supervised_learning': '监督学习',
                'unsupervised_learning': '无监督学习',
                'semi_supervised_learning': '半监督学习',
                'federated_learning': '联邦学习',
                'generative_adversarial_network': '生成对抗网络',
                'variational_autoencoder': '变分自编码器',
                'support_vector_machine': '支持向量机',
                'random_forest': '随机森林',
                'gradient_boosting': '梯度提升',
                'ensemble_learning': '集成学习',
                'feature_engineering': '特征工程',
                'hyperparameter_tuning': '超参数调优',
                'cross_validation': '交叉验证',
                'confusion_matrix': '混淆矩阵'
            },
            'ar': {
                'artificial_intelligence': 'الذكاء الاصطناعي',
                'machine_learning': 'تعلم الآلة',
                'deep_learning': 'التعلم العميق',
                'neural_network': 'الشبكة العصبية',
                'convolutional_neural_network': 'الشبكة العصبية التلافيفية',
                'recurrent_neural_network': 'الشبكة العصبية المتكررة',
                'transformer': 'المحول',
                'attention_mechanism': 'آلية الانتباه',
                'natural_language_processing': 'معالجة اللغة الطبيعية',
                'computer_vision': 'رؤية الحاسوب',
                'reinforcement_learning': 'التعلم المعزز',
                'supervised_learning': 'التعلم الإشرافي',
                'unsupervised_learning': 'التعلم غير الإشرافي',
                'semi_supervised_learning': 'التعلم شبه الإشرافي',
                'federated_learning': 'التعلم الفيدرالي',
                'generative_adversarial_network': 'الشبكة التوليدية التنافسية',
                'variational_autoencoder': 'الترميز التلقائي Variational',
                'support_vector_machine': 'آلة المتجهات الداعمة',
                'random_forest': 'الغابة العشوائية',
                'gradient_boosting': 'تعزيز التدرج',
                'ensemble_learning': 'التعلم الجماعي',
                'feature_engineering': 'هندسة الميزات',
                'hyperparameter_tuning': 'ضبط المعلمات الفائقة',
                'cross_validation': 'التحقق المتقاطع',
                'confusion_matrix': 'مصفوفة الالتباس'
            }
        }
        
        self.vocabularies['ai'] = ai_terms
    
    def _initialize_security_vocabulary(self):
        """Initialize security vocabulary."""
        security_terms = {
            'en': {
                'quantum_safe_cryptography': 'Quantum-Safe Cryptography',
                'post_quantum_cryptography': 'Post-Quantum Cryptography',
                'lattice_based_cryptography': 'Lattice-Based Cryptography',
                'hash_based_cryptography': 'Hash-Based Cryptography',
                'code_based_cryptography': 'Code-Based Cryptography',
                'multivariate_cryptography': 'Multivariate Cryptography',
                'zero_trust_security': 'Zero-Trust Security',
                'decentralized_identifier': 'Decentralized Identifier',
                'verifiable_credential': 'Verifiable Credential',
                'blockchain': 'Blockchain',
                'smart_contract': 'Smart Contract',
                'digital_signature': 'Digital Signature',
                'public_key_infrastructure': 'Public Key Infrastructure',
                'encryption': 'Encryption',
                'decryption': 'Decryption',
                'authentication': 'Authentication',
                'authorization': 'Authorization',
                'access_control': 'Access Control',
                'threat_modeling': 'Threat Modeling',
                'vulnerability_assessment': 'Vulnerability Assessment',
                'penetration_testing': 'Penetration Testing',
                'security_audit': 'Security Audit',
                'compliance': 'Compliance',
                'risk_assessment': 'Risk Assessment',
                'incident_response': 'Incident Response'
            },
            'ru': {
                'quantum_safe_cryptography': 'Квантово-безопасная криптография',
                'post_quantum_cryptography': 'Постквантовая криптография',
                'lattice_based_cryptography': 'Криптография на решетках',
                'hash_based_cryptography': 'Хэш-базированная криптография',
                'code_based_cryptography': 'Кодовая криптография',
                'multivariate_cryptography': 'Многомерная криптография',
                'zero_trust_security': 'Безопасность с нулевым доверием',
                'decentralized_identifier': 'Децентрализованный идентификатор',
                'verifiable_credential': 'Верифицируемый сертификат',
                'blockchain': 'Блокчейн',
                'smart_contract': 'Смарт-контракт',
                'digital_signature': 'Цифровая подпись',
                'public_key_infrastructure': 'Инфраструктура открытых ключей',
                'encryption': 'Шифрование',
                'decryption': 'Расшифрование',
                'authentication': 'Аутентификация',
                'authorization': 'Авторизация',
                'access_control': 'Контроль доступа',
                'threat_modeling': 'Моделирование угроз',
                'vulnerability_assessment': 'Оценка уязвимостей',
                'penetration_testing': 'Тестирование на проникновение',
                'security_audit': 'Аудит безопасности',
                'compliance': 'Соответствие',
                'risk_assessment': 'Оценка рисков',
                'incident_response': 'Реагирование на инциденты'
            },
            'zh': {
                'quantum_safe_cryptography': '抗量子密码学',
                'post_quantum_cryptography': '后量子密码学',
                'lattice_based_cryptography': '基于格的密码学',
                'hash_based_cryptography': '基于哈希的密码学',
                'code_based_cryptography': '基于编码的密码学',
                'multivariate_cryptography': '多变量密码学',
                'zero_trust_security': '零信任安全',
                'decentralized_identifier': '去中心化标识符',
                'verifiable_credential': '可验证凭证',
                'blockchain': '区块链',
                'smart_contract': '智能合约',
                'digital_signature': '数字签名',
                'public_key_infrastructure': '公钥基础设施',
                'encryption': '加密',
                'decryption': '解密',
                'authentication': '认证',
                'authorization': '授权',
                'access_control': '访问控制',
                'threat_modeling': '威胁建模',
                'vulnerability_assessment': '漏洞评估',
                'penetration_testing': '渗透测试',
                'security_audit': '安全审计',
                'compliance': '合规',
                'risk_assessment': '风险评估',
                'incident_response': '事件响应'
            },
            'ar': {
                'quantum_safe_cryptography': 'التشفير الكمي الآمن',
                'post_quantum_cryptography': 'التشفير ما بعد الكم',
                'lattice_based_cryptography': 'التشفير القائم على الشبكة',
                'hash_based_cryptography': 'التشفير القائم على التجزئة',
                'code_based_cryptography': 'التشفير القائم على الشيفرة',
                'multivariate_cryptography': 'التشفير متعدد المتغيرات',
                'zero_trust_security': 'أمان الثقة الصفرية',
                'decentralized_identifier': 'المعرف اللامركزي',
                'verifiable_credential': 'الاعتماد القابل للتحقق',
                'blockchain': 'بلوك تشين',
                'smart_contract': 'العقد الذكي',
                'digital_signature': 'التوقيع الرقمي',
                'public_key_infrastructure': 'بنية المفتاح العام',
                'encryption': 'التشفير',
                'decryption': 'فك التشفير',
                'authentication': 'المصادقة',
                'authorization': 'التفويض',
                'access_control': 'التحكم في الوصول',
                'threat_modeling': 'نمذجة التهديد',
                'vulnerability_assessment': 'تقييم الثغرات',
                'penetration_testing': 'اختبار الاختراق',
                'security_audit': 'تدقيق الأمان',
                'compliance': 'الامتثال',
                'risk_assessment': 'تقييم المخاطر',
                'incident_response': 'الاستجابة للحوادث'
            }
        }
        
        self.vocabularies['security'] = security_terms
    
    def _initialize_networking_vocabulary(self):
        """Initialize networking vocabulary."""
        networking_terms = {
            'en': {
                'quantum_mesh_protocol': 'Quantum Mesh Protocol',
                'post_dns': 'Post-DNS',
                'decentralized_network': 'Decentralized Network',
                'peer_to_peer': 'Peer-to-Peer',
                'distributed_system': 'Distributed System',
                'consensus_algorithm': 'Consensus Algorithm',
                'blockchain_network': 'Blockchain Network',
                'overlay_network': 'Overlay Network',
                'virtual_private_network': 'Virtual Private Network',
                'software_defined_network': 'Software-Defined Network',
                'network_function_virtualization': 'Network Function Virtualization',
                'content_delivery_network': 'Content Delivery Network',
                'edge_computing': 'Edge Computing',
                'fog_computing': 'Fog Computing',
                'cloud_computing': 'Cloud Computing',
                'microservices': 'Microservices',
                'containerization': 'Containerization',
                'kubernetes': 'Kubernetes',
                'docker': 'Docker',
                'api_gateway': 'API Gateway',
                'load_balancer': 'Load Balancer',
                'reverse_proxy': 'Reverse Proxy',
                'content_distribution': 'Content Distribution',
                'network_security': 'Network Security'
            },
            'ru': {
                'quantum_mesh_protocol': 'Протокол квантовой сетки',
                'post_dns': 'Post-DNS',
                'decentralized_network': 'Децентрализованная сеть',
                'peer_to_peer': 'Одноранговая сеть',
                'distributed_system': 'Распределенная система',
                'consensus_algorithm': 'Алгоритм консенсуса',
                'blockchain_network': 'Сеть блокчейн',
                'overlay_network': 'Наложенная сеть',
                'virtual_private_network': 'Виртуальная частная сеть',
                'software_defined_network': 'Программируемая сеть',
                'network_function_virtualization': 'Виртуализация сетевых функций',
                'content_delivery_network': 'Сеть доставки контента',
                'edge_computing': 'Пограничные вычисления',
                'fog_computing': 'Туманные вычисления',
                'cloud_computing': 'Облачные вычисления',
                'microservices': 'Микросервисы',
                'containerization': 'Контейнеризация',
                'kubernetes': 'Kubernetes',
                'docker': 'Docker',
                'api_gateway': 'Шлюз API',
                'load_balancer': 'Балансировщик нагрузки',
                'reverse_proxy': 'Обратный прокси',
                'content_distribution': 'Распространение контента',
                'network_security': 'Сетевая безопасность'
            },
            'zh': {
                'quantum_mesh_protocol': '量子网格协议',
                'post_dns': '后 DNS',
                'decentralized_network': '去中心化网络',
                'peer_to_peer': '点对点',
                'distributed_system': '分布式系统',
                'consensus_algorithm': '共识算法',
                'blockchain_network': '区块链网络',
                'overlay_network': '覆盖网络',
                'virtual_private_network': '虚拟专用网络',
                'software_defined_network': '软件定义网络',
                'network_function_virtualization': '网络功能虚拟化',
                'content_delivery_network': '内容分发网络',
                'edge_computing': '边缘计算',
                'fog_computing': '雾计算',
                'cloud_computing': '云计算',
                'microservices': '微服务',
                'containerization': '容器化',
                'kubernetes': 'Kubernetes',
                'docker': 'Docker',
                'api_gateway': 'API 网关',
                'load_balancer': '负载均衡器',
                'reverse_proxy': '反向代理',
                'content_distribution': '内容分发',
                'network_security': '网络安全'
            },
            'ar': {
                'quantum_mesh_protocol': 'بروتوكول الشبكة الكمية',
                'post_dns': 'Post-DNS',
                'decentralized_network': 'الشبكة اللامركزية',
                'peer_to_peer': 'الند للند',
                'distributed_system': 'النظام الموزع',
                'consensus_algorithm': 'خوارزمية الإجماع',
                'blockchain_network': 'شبكة بلوك تشين',
                'overlay_network': 'الشبكة المتراكبة',
                'virtual_private_network': 'الشبكة الافتراضية الخاصة',
                'software_defined_network': 'الشبكة المعرفة برمجياً',
                'network_function_virtualization': 'توحيد وظائف الشبكة',
                'content_delivery_network': 'شبكة توصيل المحتوى',
                'edge_computing': 'الحوسبة الحافة',
                'fog_computing': 'الحوسبة الضبابية',
                'cloud_computing': 'الحوسبة السحابية',
                'microservices': 'الخدمات المصغرة',
                'containerization': 'التحويص',
                'kubernetes': 'Kubernetes',
                'docker': 'Docker',
                'api_gateway': 'بوابة API',
                'load_balancer': 'موازن الحمل',
                'reverse_proxy': 'الوكيل العكسي',
                'content_distribution': 'توزيع المحتوى',
                'network_security': 'أمان الشبكة'
            }
        }
        
        self.vocabularies['networking'] = networking_terms
    
    def _initialize_computing_vocabulary(self):
        """Initialize general computing vocabulary."""
        computing_terms = {
            'en': {
                'quantum_infrastructure_zero': 'Quantum Infrastructure Zero',
                'hybrid_quantum_classical': 'Hybrid Quantum-Classical',
                'distributed_computing': 'Distributed Computing',
                'parallel_computing': 'Parallel Computing',
                'high_performance_computing': 'High-Performance Computing',
                'grid_computing': 'Grid Computing',
                'cluster_computing': 'Cluster Computing',
                'cloud_computing': 'Cloud Computing',
                'quantum_cloud': 'Quantum Cloud',
                'hybrid_cloud': 'Hybrid Cloud',
                'multi_cloud': 'Multi-Cloud',
                'serverless_computing': 'Serverless Computing',
                'function_as_a_service': 'Function as a Service',
                'container_as_a_service': 'Container as a Service',
                'platform_as_a_service': 'Platform as a Service',
                'infrastructure_as_a_service': 'Infrastructure as a Service',
                'software_as_a_service': 'Software as a Service',
                'virtual_machine': 'Virtual Machine',
                'virtual_environment': 'Virtual Environment',
                'computational_complexity': 'Computational Complexity',
                'algorithmic_efficiency': 'Algorithmic Efficiency',
                'computational_resources': 'Computational Resources',
                'processing_power': 'Processing Power',
                'memory_management': 'Memory Management'
            },
            'ru': {
                'quantum_infrastructure_zero': 'Квантовая инфраструктура Zero',
                'hybrid_quantum_classical': 'Гибридные квантово-классические',
                'distributed_computing': 'Распределенные вычисления',
                'parallel_computing': 'Параллельные вычисления',
                'high_performance_computing': 'Высокопроизводительные вычисления',
                'grid_computing': 'Грид-вычисления',
                'cluster_computing': 'Кластерные вычисления',
                'cloud_computing': 'Облачные вычисления',
                'quantum_cloud': 'Квантовое облако',
                'hybrid_cloud': 'Гибридное облако',
                'multi_cloud': 'Мультиоблако',
                'serverless_computing': 'Бессерверные вычисления',
                'function_as_a_service': 'Функция как сервис',
                'container_as_a_service': 'Контейнер как сервис',
                'platform_as_a_service': 'Платформа как сервис',
                'infrastructure_as_a_service': 'Инфраструктура как сервис',
                'software_as_a_service': 'Программное обеспечение как сервис',
                'virtual_machine': 'Виртуальная машина',
                'virtual_environment': 'Виртуальная среда',
                'computational_complexity': 'Вычислительная сложность',
                'algorithmic_efficiency': 'Алгоритмическая эффективность',
                'computational_resources': 'Вычислительные ресурсы',
                'processing_power': 'Вычислительная мощность',
                'memory_management': 'Управление памятью'
            },
            'zh': {
                'quantum_infrastructure_zero': '量子基础设施零',
                'hybrid_quantum_classical': '混合量子-经典',
                'distributed_computing': '分布式计算',
                'parallel_computing': '并行计算',
                'high_performance_computing': '高性能计算',
                'grid_computing': '网格计算',
                'cluster_computing': '集群计算',
                'cloud_computing': '云计算',
                'quantum_cloud': '量子云',
                'hybrid_cloud': '混合云',
                'multi_cloud': '多云',
                'serverless_computing': '无服务器计算',
                'function_as_a_service': '函数即服务',
                'container_as_a_service': '容器即服务',
                'platform_as_a_service': '平台即服务',
                'infrastructure_as_a_service': '基础设施即服务',
                'software_as_a_service': '软件即服务',
                'virtual_machine': '虚拟机',
                'virtual_environment': '虚拟环境',
                'computational_complexity': '计算复杂性',
                'algorithmic_efficiency': '算法效率',
                'computational_resources': '计算资源',
                'processing_power': '处理能力',
                'memory_management': '内存管理'
            },
            'ar': {
                'quantum_infrastructure_zero': 'البنية التحتية الكمية الصفرية',
                'hybrid_quantum_classical': 'الكمي-الكلاسيكي الهجين',
                'distributed_computing': 'الحوسبة الموزعة',
                'parallel_computing': 'الحوسبة المتوازية',
                'high_performance_computing': 'الحوسبة عالية الأداء',
                'grid_computing': 'الحوسبة الشبكية',
                'cluster_computing': 'الحوسبة العنقودية',
                'cloud_computing': 'الحوسبة السحابية',
                'quantum_cloud': 'السحابة الكمية',
                'hybrid_cloud': 'السحابة الهجينة',
                'multi_cloud': 'متعدد السحابة',
                'serverless_computing': 'الحوسبة بدون خادم',
                'function_as_a_service': 'الوظيفة كخدمة',
                'container_as_a_service': 'الحاوية كخدمة',
                'platform_as_a_service': 'المنصة كخدمة',
                'infrastructure_as_a_service': 'البنية التحتية كخدمة',
                'software_as_a_service': 'البرمجيات كخدمة',
                'virtual_machine': 'الآلة الافتراضية',
                'virtual_environment': 'البيئة الافتراضية',
                'computational_complexity': 'التعقيد الحسابي',
                'algorithmic_efficiency': 'كفاءة الخوارزمية',
                'computational_resources': 'الموارد الحسابية',
                'processing_power': 'قوة المعالجة',
                'memory_management': 'إدارة الذاكرة'
            }
        }
        
        self.vocabularies['computing'] = computing_terms
    
    def _initialize_qiz_vocabulary(self):
        """Initialize QIZ vocabulary."""
        qiz_terms = {
            'en': {
                'quantum_signature': 'Quantum Signature',
                'zero_dns_entry': 'Zero-DNS Entry',
                'zero_dns_system': 'Zero-DNS System',
                'registering_entry': 'Registering entry',
                'resolving_entry': 'Resolving entry',
                'listing_dns_entries': 'Listing DNS entries',
                'quantum_mesh_protocol': 'Quantum Mesh Protocol',
                'adding_neighbor': 'Adding neighbor',
                'removing_neighbor': 'Removing neighbor',
                'updating_routing_table': 'Updating routing table',
                'routing_message': 'Routing message',
                'no_route_to_destination': 'No route to destination',
                'zero_server': 'Zero-Server',
                'registering_service': 'Registering service',
                'getting_service': 'Getting service',
                'listing_services': 'Listing services',
                'post_dns_logic': 'Post-DNS Logic',
                'adding_logic_rule': 'Adding logic rule',
                'evaluating_rules': 'Evaluating rules',
                'rule_evaluation_failed': 'Rule evaluation failed',
                'self_contained_deploy_engine': 'Self-Contained Deploy Engine',
                'deploying_application': 'Deploying application',
                'undeploying_application': 'Undeploying application',
                'zero_trust_model': 'Zero-Trust Model',
                'adding_security_policy': 'Adding security policy',
                'validating_access': 'Validating access'
            },
            'ru': {
                'quantum_signature': 'Квантовая подпись',
                'zero_dns_entry': 'Запись Zero-DNS',
                'zero_dns_system': 'Система Zero-DNS',
                'registering_entry': 'Регистрация записи',
                'resolving_entry': 'Разрешение записи',
                'listing_dns_entries': 'Список записей DNS',
                'quantum_mesh_protocol': 'Протокол квантовой сетки',
                'adding_neighbor': 'Добавление соседа',
                'removing_neighbor': 'Удаление соседа',
                'updating_routing_table': 'Обновление таблицы маршрутизации',
                'routing_message': 'Маршрутизация сообщения',
                'no_route_to_destination': 'Нет маршрута к назначению',
                'zero_server': 'Zero-Сервер',
                'registering_service': 'Регистрация сервиса',
                'getting_service': 'Получение сервиса',
                'listing_services': 'Список сервисов',
                'post_dns_logic': 'Логика Post-DNS',
                'adding_logic_rule': 'Добавление правила логики',
                'evaluating_rules': 'Оценка правил',
                'rule_evaluation_failed': 'Ошибка оценки правила',
                'self_contained_deploy_engine': 'Автономный движок развертывания',
                'deploying_application': 'Развертывание приложения',
                'undeploying_application': 'Отмена развертывания приложения',
                'zero_trust_model': 'Модель Zero-Trust',
                'adding_security_policy': 'Добавление политики безопасности',
                'validating_access': 'Проверка доступа'
            },
            'zh': {
                'quantum_signature': '量子签名',
                'zero_dns_entry': 'Zero-DNS 条目',
                'zero_dns_system': 'Zero-DNS 系统',
                'registering_entry': '注册条目',
                'resolving_entry': '解析条目',
                'listing_dns_entries': '列出 DNS 条目',
                'quantum_mesh_protocol': '量子网格协议',
                'adding_neighbor': '添加邻居',
                'removing_neighbor': '移除邻居',
                'updating_routing_table': '更新路由表',
                'routing_message': '路由消息',
                'no_route_to_destination': '无路由到目的地',
                'zero_server': 'Zero-服务器',
                'registering_service': '注册服务',
                'getting_service': '获取服务',
                'listing_services': '列出服务',
                'post_dns_logic': 'Post-DNS 逻辑',
                'adding_logic_rule': '添加逻辑规则',
                'evaluating_rules': '评估规则',
                'rule_evaluation_failed': '规则评估失败',
                'self_contained_deploy_engine': '自包含部署引擎',
                'deploying_application': '部署应用',
                'undeploying_application': '取消部署应用',
                'zero_trust_model': '零信任模型',
                'adding_security_policy': '添加安全策略',
                'validating_access': '验证访问'
            },
            'ar': {
                'quantum_signature': 'التوقيع الكمي',
                'zero_dns_entry': 'إدخال Zero-DNS',
                'zero_dns_system': 'نظام Zero-DNS',
                'registering_entry': 'تسجيل الإدخال',
                'resolving_entry': 'حل الإدخال',
                'listing_dns_entries': 'إدراج إدخالات DNS',
                'quantum_mesh_protocol': 'بروتوكول الشبكة الكمية',
                'adding_neighbor': 'إضافة جار',
                'removing_neighbor': 'إزالة جار',
                'updating_routing_table': 'تحديث جدول التوجيه',
                'routing_message': 'توجيه الرسالة',
                'no_route_to_destination': 'لا يوجد مسار إلى الوجهة',
                'zero_server': 'خادم Zero',
                'registering_service': 'تسجيل الخدمة',
                'getting_service': 'الحصول على الخدمة',
                'listing_services': 'إدراج الخدمات',
                'post_dns_logic': 'منطق Post-DNS',
                'adding_logic_rule': 'إضافة قاعدة منطقية',
                'evaluating_rules': 'تقييم القواعد',
                'rule_evaluation_failed': 'فشل تقييم القاعدة',
                'self_contained_deploy_engine': 'محرك النشر الذاتي',
                'deploying_application': 'نشر التطبيق',
                'undeploying_application': 'إلغاء نشر التطبيق',
                'zero_trust_model': 'نموذج الثقة الصفرية',
                'adding_security_policy': 'إضافة سياسة أمان',
                'validating_access': 'التحقق من الوصول'
            }
        }
        
        self.vocabularies['qiz'] = qiz_terms
    
    def translate_term(self, term: str, domain: str, target_language: str, source_language: str = 'en') -> str:
        """
        Translate a technical term from source language to target language in a specific domain.
        
        Args:
            term (str): Technical term to translate
            domain (str): Domain of the term (quantum, ai, security, etc.)
            target_language (str): Target language code
            source_language (str): Source language code (default: 'en')
            
        Returns:
            str: Translated term or original term if not found
        """
        # Check if domain exists
        if domain not in self.vocabularies:
            return term
        
        # Check if target language exists
        if target_language not in self.vocabularies[domain]:
            return term
        
        # Find term in source language
        source_terms = self.vocabularies[domain].get(source_language, {})
        target_terms = self.vocabularies[domain].get(target_language, {})
        
        # Find matching term
        for key, value in source_terms.items():
            if value.lower() == term.lower():
                # Return translation if found
                if key in target_terms:
                    return target_terms[key]
                break
        
        # If exact match not found, try partial matching
        for key, value in source_terms.items():
            if term.lower() in value.lower() or value.lower() in term.lower():
                if key in target_terms:
                    return target_terms[key]
        
        # Return original term if not found
        return term
    
    def get_domain_terms(self, domain: str, language: str = 'en') -> Dict[str, str]:
        """
        Get all terms for a specific domain in a specific language.
        
        Args:
            domain (str): Domain name
            language (str): Language code (default: 'en')
            
        Returns:
            dict: Dictionary of terms in the specified language
        """
        if domain in self.vocabularies:
            return self.vocabularies[domain].get(language, {})
        return {}
    
    def get_supported_domains(self) -> List[str]:
        """
        Get list of supported domains.
        
        Returns:
            list: List of supported domains
        """
        return list(self.vocabularies.keys())
    
    def get_supported_languages(self) -> List[str]:
        """
        Get list of supported languages.
        
        Returns:
            list: List of supported languages
        """
        return self.supported_languages.copy()
    
    def add_term(self, term_key: str, translations: Dict[str, str], domain: str):
        """
        Add a new technical term to the vocabulary.
        
        Args:
            term_key (str): Unique key for the term
            translations (dict): Dictionary of language codes to translations
            domain (str): Domain for the term
        """
        if domain not in self.vocabularies:
            self.vocabularies[domain] = {}
        
        for language in self.supported_languages:
            if language not in self.vocabularies[domain]:
                self.vocabularies[domain][language] = {}
            if term_key in translations:
                self.vocabularies[domain][language][term_key] = translations[term_key]
    
    def find_terms_in_text(self, text: str, domain: str, language: str = 'en') -> List[str]:
        """
        Find technical terms in a text for a specific domain and language.
        
        Args:
            text (str): Text to search for terms
            domain (str): Domain to search in
            language (str): Language of the text (default: 'en')
            
        Returns:
            list: List of found technical terms
        """
        found_terms = []
        
        if domain in self.vocabularies:
            terms = self.vocabularies[domain].get(language, {})
            
            # Search for terms in text
            for term_key, term_value in terms.items():
                if term_value.lower() in text.lower():
                    found_terms.append(term_value)
        
        return found_terms


# Global instance
_vocabulary_manager = None


def get_vocabulary_manager() -> VocabularyManager:
    """
    Get the global vocabulary manager instance.
    
    Returns:
        VocabularyManager: Global vocabulary manager instance
    """
    global _vocabulary_manager
    if _vocabulary_manager is None:
        _vocabulary_manager = VocabularyManager()
    return _vocabulary_manager