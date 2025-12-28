// Multilingual support for AIPlatform website
const translations = {
    en: {
        nav: {
            home: "Home",
            features: "Features",
            documentation: "Documentation",
            examples: "Examples",
            demo: "Demo",
            community: "Community",
            github: "GitHub",
            get_started: "Get Started"
        },
        hero: {
            title: "Quantum-AI Infrastructure<br><span class=\"highlight\">Zero SDK</span>",
            subtitle: "The next-generation SDK for quantum computing, federated AI, computer vision,<br>and zero-infrastructure networking. Built for the Web6 era.",
            get_started: "Get Started",
            view_demo: "View Demo",
            stats: {
                modules: "10+",
                modules_label: "Core Modules",
                algorithms: "50+",
                algorithms_label: "Quantum Algorithms",
                examples: "100+",
                examples_label: "Examples"
            }
        },
        features: {
            title: "Revolutionary Features",
            subtitle: "Built for the future of quantum-AI integration",
            quantum: {
                title: "Quantum Computing",
                description: "Full Qiskit Runtime integration with support for IBM Nighthawk & Heron QPUs, quantum algorithms (VQE, QAOA, Grover, Shor), and quantum-safe cryptography."
            },
            zero: {
                title: "Zero Infrastructure",
                description: "Quantum Infrastructure Zero (QIZ) with no servers, no DNS, no routes. Post-DNS interaction layer and Quantum Mesh Protocol (QMP)."
            },
            federated: {
                title: "Federated Quantum AI",
                description: "Distributed training between nodes with hybrid quantum-classical learning, model marketplace, and NFT-based model weights."
            },
            vision: {
                title: "Vision & Data Lab",
                description: "Computer vision, 3D processing, multimodal AI, WebXR integration, big data pipelines, and streaming analytics."
            },
            genai: {
                title: "GenAI Integration",
                description: "Support for OpenAI, Claude, LLaMA, GigaChat3-702B, Katya AI, diffusion models, and MCP protocol interactions."
            },
            security: {
                title: "Quantum Security",
                description: "Quantum-safe cryptography (Kyber, Dilithium), Zero-Trust model, and DIDN implementation."
            }
        },
        ibm: {
            title: "Enterprise Quantum Integration",
            subtitle: "Powered by IBM Quantum with Qiskit Runtime, Quantum Safe Crypto, IBM Nighthawk & Heron support, and IBM Hybrid Cloud integration.",
            qiskit: {
                title: "Qiskit Runtime",
                description: "Seamless integration with IBM's quantum computing platform"
            },
            crypto: {
                title: "Quantum Safe Crypto",
                description: "Cryptography resistant to quantum computer attacks"
            },
            cloud: {
                title: "Hybrid Cloud",
                description: "Enterprise-grade deployment across quantum and classical infrastructure"
            }
        },
        revolution: {
            title: "AI Quantum Revolution",
            subtitle: "From Web 3 to Web 6: Multiplatform quantum engineering, meta-consciousness, and DAO governance.",
            web3: {
                title: "Web 3 Foundation",
                description: "Decentralized web with blockchain and smart contracts"
            },
            web4: {
                title: "Web 4 Evolution",
                description: "Semantic web with AI-powered content understanding"
            },
            web5: {
                title: "Web 5 Integration",
                description: "Quantum-AI hybrid systems with distributed intelligence"
            },
            web6: {
                title: "Web 6 Revolution",
                description: "Meta-consciousness platforms with quantum neural networks"
            }
        },
        docs: {
            title: "Comprehensive Documentation",
            subtitle: "Everything you need to build with AIPlatform",
            quick_start: {
                title: "Quick Start Guide",
                description: "Get up and running with AIPlatform in minutes",
                link: "Read Guide →"
            },
            quantum: {
                title: "Quantum Integration",
                description: "Complete guide to quantum computing integration",
                link: "Read Guide →"
            },
            vision: {
                title: "Vision Module API",
                description: "Computer vision and data processing APIs",
                link: "Read API →"
            },
            federated: {
                title: "Federated Training",
                description: "Distributed AI training with quantum acceleration",
                link: "Read Manual →"
            },
            qiz: {
                title: "Web6 & QIZ Architecture",
                description: "Zero-infrastructure networking and protocols",
                link: "Read Architecture →"
            },
            api: {
                title: "API Reference",
                description: "Complete API documentation and examples",
                link: "View Reference →"
            }
        },
        examples: {
            title: "Working Examples",
            subtitle: "Learn by doing with real-world examples",
            quantum: {
                title: "Quantum + AI Hybrid",
                description: "Model with quantum processing unit and GPU components",
                link: "View Code →"
            },
            vision: {
                title: "Vision Recognition",
                description: "Object detection, face recognition, and gesture processing",
                link: "View Code →"
            },
            federated: {
                title: "Federated Training",
                description: "Distributed model training across quantum nodes",
                link: "View Code →"
            },
            multimodal: {
                title: "Multimodal AI",
                description: "Text, audio, video, and 3D processing integration",
                link: "View Code →"
            }
        },
        demo: {
            title: "Interactive Demo",
            subtitle: "Experience the power of quantum-AI integration",
            placeholder: "AIPlatform Quantum Revolution Demo",
            description: {
                title: "Quantum-AI Platform Demo",
                subtitle: "Watch as we demonstrate the integration of quantum computing with artificial intelligence, showcasing:"
            },
            features: {
                qpu: "Quantum circuit execution on IBM QPUs",
                federated: "Federated learning across distributed nodes",
                vision: "Computer vision processing with 3D object recognition",
                qiz: "Zero-infrastructure deployment with QIZ",
                multimodal: "Multimodal AI processing with GigaChat3-702B"
            },
            download: "Download Demo"
        },
        community: {
            title: "Join Our Community",
            subtitle: "Connect with quantum-AI developers and researchers",
            github: {
                title: "GitHub",
                description: "Contribute to the open-source project",
                link: "View Repository →"
            },
            docs: {
                title: "Documentation",
                description: "Comprehensive guides and API references",
                link: "View Docs →"
            },
            papers: {
                title: "Research Papers",
                description: "Whitepapers on quantum infrastructure and AI",
                link: "Read Papers →"
            },
            support: {
                title: "Support",
                description: "Get help from our developer community",
                link: "Get Support →"
            }
        },
        footer: {
            product: {
                title: "Product",
                features: "Features",
                documentation: "Documentation",
                examples: "Examples",
                demo: "Demo"
            },
            resources: {
                title: "Resources",
                api: "API Reference",
                papers: "Whitepapers",
                blog: "Blog",
                research: "Research"
            },
            company: {
                title: "Company",
                about: "About",
                careers: "Careers",
                contact: "Contact",
                partners: "Partners"
            },
            copyright: "© 2025 REChain Network Solutions & Katya AI Systems. All rights reserved.",
            social: {
                twitter: "Twitter",
                linkedin: "LinkedIn",
                github: "GitHub"
            }
        }
    },
    ru: {
        nav: {
            home: "Главная",
            features: "Возможности",
            documentation: "Документация",
            examples: "Примеры",
            demo: "Демо",
            community: "Сообщество",
            github: "GitHub",
            get_started: "Начать"
        },
        hero: {
            title: "Квантово-ИИ Инфраструктура<br><span class=\"highlight\">Zero SDK</span>",
            subtitle: "Следующее поколение SDK для квантовых вычислений, федеративного ИИ, компьютерного зрения<br>и сетей с нулевой инфраструктурой. Создано для эпохи Web6.",
            get_started: "Начать",
            view_demo: "Посмотреть Демо",
            stats: {
                modules: "10+",
                modules_label: "Основных Модулей",
                algorithms: "50+",
                algorithms_label: "Квантовых Алгоритмов",
                examples: "100+",
                examples_label: "Примеров"
            }
        },
        features: {
            title: "Революционные Возможности",
            subtitle: "Создано для будущего квантово-ИИ интеграции",
            quantum: {
                title: "Квантовые Вычисления",
                description: "Полная интеграция Qiskit Runtime с поддержкой IBM Nighthawk & Heron QPU, квантовые алгоритмы (VQE, QAOA, Grover, Shor) и квантово-безопасная криптография."
            },
            zero: {
                title: "Нулевая Инфраструктура",
                description: "Квантовая Инфраструктура Zero (QIZ) без серверов, DNS, маршрутов. Пост-DNS слой взаимодействия и Квантовый Сетевой Протокол (QMP)."
            },
            federated: {
                title: "Федеративный Квантовый ИИ",
                description: "Распределенное обучение между узлами с гибридным квантово-классическим обучением, рынок моделей и веса моделей на основе NFT."
            },
            vision: {
                title: "Лаборатория Зрения и Данных",
                description: "Компьютерное зрение, 3D-обработка, мультимодальный ИИ, интеграция WebXR, конвейеры больших данных и потоковая аналитика."
            },
            genai: {
                title: "Интеграция Генеративного ИИ",
                description: "Поддержка OpenAI, Claude, LLaMA, GigaChat3-702B, Katya AI, диффузионные модели и взаимодействия протокола MCP."
            },
            security: {
                title: "Квантовая Безопасность",
                description: "Квантово-безопасная криптография (Kyber, Dilithium), модель Zero-Trust и реализация DIDN."
            }
        },
        ibm: {
            title: "Корпоративная Квантовая Интеграция",
            subtitle: "На базе IBM Quantum с Qiskit Runtime, Quantum Safe Crypto, поддержкой IBM Nighthawk & Heron и интеграцией IBM Hybrid Cloud.",
            qiskit: {
                title: "Qiskit Runtime",
                description: "Бесшовная интеграция с квантовой вычислительной платформой IBM"
            },
            crypto: {
                title: "Квантово-безопасная Криптография",
                description: "Криптография, устойчивая к атакам квантовых компьютеров"
            },
            cloud: {
                title: "Гибридное Облако",
                description: "Корпоративный уровень развертывания в квантовой и классической инфраструктуре"
            }
        },
        revolution: {
            title: "Квантовая Революция ИИ",
            subtitle: "От Web 3 к Web 6: Мультиплатформенная квантовая инженерия, мета-сознание и DAO-управление.",
            web3: {
                title: "Основа Web 3",
                description: "Децентрализованный веб с блокчейном и смарт-контрактами"
            },
            web4: {
                title: "Эволюция Web 4",
                description: "Семантический веб с пониманием контента на базе ИИ"
            },
            web5: {
                title: "Интеграция Web 5",
                description: "Гибридные квантово-ИИ системы с распределенным интеллектом"
            },
            web6: {
                title: "Революция Web 6",
                description: "Платформы мета-сознания с квантовыми нейронными сетями"
            }
        },
        docs: {
            title: "Полная Документация",
            subtitle: "Все, что нужно для разработки с AIPlatform",
            quick_start: {
                title: "Руководство Быстрого Старта",
                description: "Начните работу с AIPlatform за считанные минуты",
                link: "Читать Руководство →"
            },
            quantum: {
                title: "Интеграция Квантовых Технологий",
                description: "Полное руководство по интеграции квантовых вычислений",
                link: "Читать Руководство →"
            },
            vision: {
                title: "API Модуля Зрения",
                description: "API компьютерного зрения и обработки данных",
                link: "Читать API →"
            },
            federated: {
                title: "Федеративное Обучение",
                description: "Распределенное обучение ИИ с квантовым ускорением",
                link: "Читать Руководство →"
            },
            qiz: {
                title: "Архитектура Web6 и QIZ",
                description: "Сети с нулевой инфраструктурой и протоколы",
                link: "Читать Архитектуру →"
            },
            api: {
                title: "Справочник API",
                description: "Полная документация API и примеры",
                link: "Просмотреть Справочник →"
            }
        },
        examples: {
            title: "Рабочие Примеры",
            subtitle: "Учитесь на практике с реальными примерами",
            quantum: {
                title: "Гибрид Квантовых Технологий + ИИ",
                description: "Модель с квантовым процессором и компонентами GPU",
                link: "Просмотреть Код →"
            },
            vision: {
                title: "Распознавание Зрения",
                description: "Обнаружение объектов, распознавание лиц и жестов",
                link: "Просмотреть Код →"
            },
            federated: {
                title: "Федеративное Обучение",
                description: "Распределенное обучение моделей по квантовым узлам",
                link: "Просмотреть Код →"
            },
            multimodal: {
                title: "Мультимодальный ИИ",
                description: "Интеграция текста, аудио, видео и 3D-обработки",
                link: "Просмотреть Код →"
            }
        },
        demo: {
            title: "Интерактивное Демо",
            subtitle: "Испытайте мощь квантово-ИИ интеграции",
            placeholder: "Демо Квантовой Революции AIPlatform",
            description: {
                title: "Демо Платформы Квантового ИИ",
                subtitle: "Смотрите, как мы демонстрируем интеграцию квантовых вычислений с искусственным интеллектом:"
            },
            features: {
                qpu: "Выполнение квантовых схем на IBM QPU",
                federated: "Федеративное обучение по распределенным узлам",
                vision: "Обработка компьютерного зрения с 3D-распознаванием объектов",
                qiz: "Развертывание с нулевой инфраструктурой с QIZ",
                multimodal: "Мультимодальная обработка ИИ с GigaChat3-702B"
            },
            download: "Скачать Демо"
        },
        community: {
            title: "Присоединяйтесь к Нашему Сообществу",
            subtitle: "Связывайтесь с разработчиками и исследователями квантового ИИ",
            github: {
                title: "GitHub",
                description: "Внесите вклад в проект с открытым исходным кодом",
                link: "Просмотреть Репозиторий →"
            },
            docs: {
                title: "Документация",
                description: "Полные руководства и справочники API",
                link: "Просмотреть Документацию →"
            },
            papers: {
                title: "Научные Статьи",
                description: "Белые книги по квантовой инфраструктуре и ИИ",
                link: "Читать Статьи →"
            },
            support: {
                title: "Поддержка",
                description: "Получите помощь от нашего сообщества разработчиков",
                link: "Получить Поддержку →"
            }
        },
        footer: {
            product: {
                title: "Продукт",
                features: "Возможности",
                documentation: "Документация",
                examples: "Примеры",
                demo: "Демо"
            },
            resources: {
                title: "Ресурсы",
                api: "Справочник API",
                papers: "Белые книги",
                blog: "Блог",
                research: "Исследования"
            },
            company: {
                title: "Компания",
                about: "О нас",
                careers: "Карьера",
                contact: "Контакты",
                partners: "Партнеры"
            },
            copyright: "© 2025 REChain Network Solutions & Katya AI Systems. Все права защищены.",
            social: {
                twitter: "Twitter",
                linkedin: "LinkedIn",
                github: "GitHub"
            }
        }
    },
    zh: {
        nav: {
            home: "首页",
            features: "功能",
            documentation: "文档",
            examples: "示例",
            demo: "演示",
            community: "社区",
            github: "GitHub",
            get_started: "开始使用"
        },
        hero: {
            title: "量子AI基础设施<br><span class=\"highlight\">零SDK</span>",
            subtitle: "下一代SDK，用于量子计算、联邦AI、计算机视觉<br>和零基础设施网络。为Web6时代而构建。",
            get_started: "开始使用",
            view_demo: "查看演示",
            stats: {
                modules: "10+",
                modules_label: "核心模块",
                algorithms: "50+",
                algorithms_label: "量子算法",
                examples: "100+",
                examples_label: "示例"
            }
        },
        features: {
            title: "革命性功能",
            subtitle: "为量子AI集成的未来而构建",
            quantum: {
                title: "量子计算",
                description: "完整的Qiskit Runtime集成，支持IBM Nighthawk和Heron QPU，量子算法（VQE、QAOA、Grover、Shor）和量子安全密码学。"
            },
            zero: {
                title: "零基础设施",
                description: "量子基础设施零（QIZ），无服务器、无DNS、无路由。后DNS交互层和量子网格协议（QMP）。"
            },
            federated: {
                title: "联邦量子AI",
                description: "节点间的分布式训练，混合量子-经典学习，模型市场和基于NFT的模型权重。"
            },
            vision: {
                title: "视觉与数据实验室",
                description: "计算机视觉、3D处理、多模态AI、WebXR集成、大数据管道和流分析。"
            },
            genai: {
                title: "生成式AI集成",
                description: "支持OpenAI、Claude、LLaMA、GigaChat3-702B、Katya AI、扩散模型和MCP协议交互。"
            },
            security: {
                title: "量子安全",
                description: "量子安全密码学（Kyber、Dilithium）、零信任模型和DIDN实现。"
            }
        },
        ibm: {
            title: "企业级量子集成",
            subtitle: "由IBM Quantum提供支持，具有Qiskit Runtime、量子安全加密、IBM Nighthawk和Heron支持以及IBM混合云集成。",
            qiskit: {
                title: "Qiskit Runtime",
                description: "与IBM量子计算平台的无缝集成"
            },
            crypto: {
                title: "量子安全加密",
                description: "抗量子计算机攻击的密码学"
            },
            cloud: {
                title: "混合云",
                description: "跨量子和经典基础设施的企业级部署"
            }
        },
        revolution: {
            title: "AI量子革命",
            subtitle: "从Web 3到Web 6：多平台量子工程、元意识和DAO治理。",
            web3: {
                title: "Web 3基础",
                description: "具有区块链和智能合约的去中心化网络"
            },
            web4: {
                title: "Web 4演进",
                description: "具有AI驱动内容理解的语义网络"
            },
            web5: {
                title: "Web 5集成",
                description: "具有分布式智能的量子AI混合系统"
            },
            web6: {
                title: "Web 6革命",
                description: "具有量子神经网络的元意识平台"
            }
        },
        docs: {
            title: "全面文档",
            subtitle: "构建AIPlatform所需的一切",
            quick_start: {
                title: "快速入门指南",
                description: "在几分钟内开始使用AIPlatform",
                link: "阅读指南 →"
            },
            quantum: {
                title: "量子集成",
                description: "量子计算集成完整指南",
                link: "阅读指南 →"
            },
            vision: {
                title: "视觉模块API",
                description: "计算机视觉和数据处理API",
                link: "阅读API →"
            },
            federated: {
                title: "联邦训练",
                description: "具有量子加速的分布式AI训练",
                link: "阅读手册 →"
            },
            qiz: {
                title: "Web6和QIZ架构",
                description: "零基础设施网络和协议",
                link: "阅读架构 →"
            },
            api: {
                title: "API参考",
                description: "完整的API文档和示例",
                link: "查看参考 →"
            }
        },
        examples: {
            title: "工作示例",
            subtitle: "通过实际示例学习",
            quantum: {
                title: "量子+AI混合",
                description: "具有量子处理单元和GPU组件的模型",
                link: "查看代码 →"
            },
            vision: {
                title: "视觉识别",
                description: "物体检测、人脸识别和手势处理",
                link: "查看代码 →"
            },
            federated: {
                title: "联邦训练",
                description: "跨量子节点的分布式模型训练",
                link: "查看代码 →"
            },
            multimodal: {
                title: "多模态AI",
                description: "文本、音频、视频和3D处理集成",
                link: "查看代码 →"
            }
        },
        demo: {
            title: "交互式演示",
            subtitle: "体验量子AI集成的强大功能",
            placeholder: "AIPlatform量子革命演示",
            description: {
                title: "量子AI平台演示",
                subtitle: "观看我们演示量子计算与人工智能的集成，展示："
            },
            features: {
                qpu: "IBM QPU上的量子电路执行",
                federated: "跨分布式节点的联邦学习",
                vision: "具有3D物体识别的计算机视觉处理",
                qiz: "使用QIZ的零基础设施部署",
                multimodal: "使用GigaChat3-702B的多模态AI处理"
            },
            download: "下载演示"
        },
        community: {
            title: "加入我们的社区",
            subtitle: "与量子AI开发者和研究人员联系",
            github: {
                title: "GitHub",
                description: "为开源项目做贡献",
                link: "查看仓库 →"
            },
            docs: {
                title: "文档",
                description: "全面的指南和API参考",
                link: "查看文档 →"
            },
            papers: {
                title: "研究论文",
                description: "量子基础设施和AI白皮书",
                link: "阅读论文 →"
            },
            support: {
                title: "支持",
                description: "从我们的开发者社区获得帮助",
                link: "获取支持 →"
            }
        },
        footer: {
            product: {
                title: "产品",
                features: "功能",
                documentation: "文档",
                examples: "示例",
                demo: "演示"
            },
            resources: {
                title: "资源",
                api: "API参考",
                papers: "白皮书",
                blog: "博客",
                research: "研究"
            },
            company: {
                title: "公司",
                about: "关于",
                careers: "职业",
                contact: "联系",
                partners: "合作伙伴"
            },
            copyright: "© 2025 REChain Network Solutions & Katya AI Systems。保留所有权利。",
            social: {
                twitter: "Twitter",
                linkedin: "LinkedIn",
                github: "GitHub"
            }
        }
    },
    ar: {
        nav: {
            home: "الرئيسية",
            features: "الميزات",
            documentation: "التوثيق",
            examples: "أمثلة",
            demo: "عرض توضيحي",
            community: "المجتمع",
            github: "GitHub",
            get_started: "ابدأ الآن"
        },
        hero: {
            title: "البنية التحتية للذكاء الاصطناعي الكمي<br><span class=\"highlight\">SDK الصفري</span>",
            subtitle: "SDK الجيل القادم للحوسبة الكمية، الذكاء الاصطناعي الفيدرالي، رؤية الكمبيوتر<br>والشبكات بدون بنية تحتية. مبني لعصر Web6.",
            get_started: "ابدأ الآن",
            view_demo: "عرض العرض التوضيحي",
            stats: {
                modules: "10+",
                modules_label: "الوحدات الأساسية",
                algorithms: "50+",
                algorithms_label: "خوارزميات كمومية",
                examples: "100+",
                examples_label: "أمثلة"
            }
        },
        features: {
            title: "ميزات ثورية",
            subtitle: "مبني لمستقبل تكامل الذكاء الاصطناعي الكمي",
            quantum: {
                title: "الحوسبة الكمية",
                description: "تكامل كامل لـ Qiskit Runtime مع دعم لـ IBM Nighthawk و Heron QPU، خوارزميات كمية (VQE، QAOA، Grover، Shor)، وتشفير آمن كمومياً."
            },
            zero: {
                title: "البنية التحتية الصفرية",
                description: "البنية التحتية الكمية الصفرية (QIZ) بدون خوادم، بدون DNS، بدون مسارات. طبقة تفاعل ما بعد DNS وبروتوكول الشبكة الكمية (QMP)."
            },
            federated: {
                title: "الذكاء الاصطناعي الكمي الفيدرالي",
                description: "التدريب الموزع بين العقد مع التعلم الهجين الكمي-الكلاسيكي، سوق النماذج، وأوزان النماذج القائمة على NFT."
            },
            vision: {
                title: "مختبر الرؤية والبيانات",
                description: "رؤية الكمبيوتر، المعالجة ثلاثية الأبعاد، الذكاء الاصطناعي متعدد الوسائط، تكامل WebXR، خطوط أنابيب البيانات الكبيرة، والتحليلات البثية."
            },
            genai: {
                title: "تكامل الذكاء الاصطناعي التوليدي",
                description: "دعم OpenAI، Claude، LLaMA، GigaChat3-702B، Katya AI، النماذج التفاضلية، وتفاعلات بروتوكول MCP."
            },
            security: {
                title: "الأمان الكمي",
                description: "التشفير الآمن كمومياً (Kyber، Dilithium)، نموذج الثقة الصفرية، وتنفيذ DIDN."
            }
        },
        ibm: {
            title: "التكامل الكمي المؤسسي",
            subtitle: "مدعوم من IBM Quantum مع Qiskit Runtime، التشفير الآمن الكمي، دعم IBM Nighthawk و Heron، وتكامل IBM Hybrid Cloud.",
            qiskit: {
                title: "Qiskit Runtime",
                description: "تكامل سلس مع منصة الحوسبة الكمية من IBM"
            },
            crypto: {
                title: "التشفير الآمن كمومياً",
                description: "تشفير مقاوم لهجمات الحواسيب الكمومية"
            },
            cloud: {
                title: "السحابة الهجينة",
                description: "نشر على مستوى المؤسسة عبر البنية التحتية الكمية والكلاسيكية"
            }
        },
        revolution: {
            title: "ثورة الذكاء الاصطناعي الكمي",
            subtitle: "من Web 3 إلى Web 6: الهندسة الكمية متعددة المنصات، الوعي الفوقى، والحوكمة القائمة على DAO.",
            web3: {
                title: "أساس Web 3",
                description: "الويب اللامركزي مع البلوك تشين والعقود الذكية"
            },
            web4: {
                title: "تطور Web 4",
                description: "الويب الدلالي مع فهم المحتوى المدعوم بالذكاء الاصطناعي"
            },
            web5: {
                title: "تكامل Web 5",
                description: "أنظمة هجينة كمية-ذكاء اصطناعي مع ذكاء موزع"
            },
            web6: {
                title: "ثورة Web 6",
                description: "منصات الوعي الفوقى مع الشبكات العصبية الكمومية"
            }
        },
        docs: {
            title: "توثيق شامل",
            subtitle: "كل ما تحتاجه للبناء مع AIPlatform",
            quick_start: {
                title: "دليل البدء السريع",
                description: "ابدأ العمل مع AIPlatform في دقائق",
                link: "اقرأ الدليل ←"
            },
            quantum: {
                title: "تكامل الكم",
                description: "دليل كامل لتكامل الحوسبة الكمية",
                link: "اقرأ الدليل ←"
            },
            vision: {
                title: "API وحدة الرؤية",
                description: "API رؤية الكمبيوتر ومعالجة البيانات",
                link: "اقرأ API ←"
            },
            federated: {
                title: "التدريب الفيدرالي",
                description: "تدريب الذكاء الاصطناعي الموزع مع تسريع كمي",
                link: "اقرأ الدليل ←"
            },
            qiz: {
                title: "بنية Web6 و QIZ",
                description: "الشبكات بدون بنية تحتية والبروتوكولات",
                link: "اقرأ البنية ←"
            },
            api: {
                title: "مرجع API",
                description: "توثيق API كامل وأمثلة",
                link: "اعرض المرجع ←"
            }
        },
        examples: {
            title: "أمثلة عملية",
            subtitle: "تعلم بالفعل مع أمثلة من العالم الحقيقي",
            quantum: {
                title: "هجين الكم + الذكاء الاصطناعي",
                description: "نموذج بوحدة معالجة كمية ومكونات GPU",
                link: "اعرض الكود ←"
            },
            vision: {
                title: "التعرف على الرؤية",
                description: "كشف الكائنات، التعرف على الوجوه، ومعالجة الإيماءات",
                link: "اعرض الكود ←"
            },
            federated: {
                title: "التدريب الفيدرالي",
                description: "تدريب النموذج الموزع عبر العقد الكمية",
                link: "اعرض الكود ←"
            },
            multimodal: {
                title: "الذكاء الاصطناعي متعدد الوسائط",
                description: "تكامل النص، الصوت، الفيديو، والمعالجة ثلاثية الأبعاد",
                link: "اعرض الكود ←"
            }
        },
        demo: {
            title: "عرض توضيحي تفاعلي",
            subtitle: "اختبر قوة تكامل الذكاء الاصطناعي الكمي",
            placeholder: "عرض ثورة الذكاء الاصطناعي الكمي AIPlatform",
            description: {
                title: "عرض منصة الذكاء الاصطناعي الكمي",
                subtitle: "شاهد كيف نُظهر تكامل الحوسبة الكمية مع الذكاء الاصطناعي، موضحاً:"
            },
            features: {
                qpu: "تنفيذ الدوائر الكمية على QPU من IBM",
                federated: "التعلم الفيدرالي عبر العقد الموزعة",
                vision: "معالجة رؤية الكمبيوتر مع التعرف على الكائنات ثلاثية الأبعاد",
                qiz: "النشر بدون بنية تحتية مع QIZ",
                multimodal: "معالجة الذكاء الاصطناعي متعدد الوسائط مع GigaChat3-702B"
            },
            download: "تحميل العرض التوضيحي"
        },
        community: {
            title: "انضم إلى مجتمعنا",
            subtitle: "تواصل مع مطوري الذكاء الاصطناعي الكمي والباحثين",
            github: {
                title: "GitHub",
                description: "ساهم في المشروع مفتوح المصدر",
                link: "اعرض المستودع ←"
            },
            docs: {
                title: "التوثيق",
                description: "أدلة شاملة ومراجع API",
                link: "اعرض التوثيق ←"
            },
            papers: {
                title: "الأوراق البحثية",
                description: "الكتب البيضاء حول البنية التحتية الكمية والذكاء الاصطناعي",
                link: "اقرأ الأوراق ←"
            },
            support: {
                title: "الدعم",
                description: "احصل على المساعدة من مجتمع المطورين لدينا",
                link: "احصل على الدعم ←"
            }
        },
        footer: {
            product: {
                title: "المنتج",
                features: "الميزات",
                documentation: "التوثيق",
                examples: "أمثلة",
                demo: "عرض توضيحي"
            },
            resources: {
                title: "الموارد",
                api: "مرجع API",
                papers: "الكتب البيضاء",
                blog: "المدونة",
                research: "البحث"
            },
            company: {
                title: "الشركة",
                about: "عن الشركة",
                careers: "الوظائف",
                contact: "اتصل بنا",
                partners: "الشركاء"
            },
            copyright: "© 2025 REChain Network Solutions & Katya AI Systems. جميع الحقوق محفوظة.",
            social: {
                twitter: "Twitter",
                linkedin: "LinkedIn",
                github: "GitHub"
            }
        }
    }
};

// Function to change language
function changeLanguage(lang) {
    // Update all elements with data-i18n attribute
    const elements = document.querySelectorAll('[data-i18n]');
    elements.forEach(element => {
        const key = element.getAttribute('data-i18n');
        const translation = getTranslation(key, lang);
        if (translation) {
            if (element.tagName === 'INPUT' || element.tagName === 'TEXTAREA') {
                element.placeholder = translation;
            } else {
                element.innerHTML = translation;
            }
        }
    });
    
    // Update language selector
    document.getElementById('language-select').value = lang;
    
    // Save language preference
    localStorage.setItem('preferredLanguage', lang);
    
    // Update text direction for Arabic
    document.documentElement.setAttribute('dir', lang === 'ar' ? 'rtl' : 'ltr');
    document.documentElement.setAttribute('lang', lang);
}

// Function to get translation by key
function getTranslation(key, lang) {
    const keys = key.split('.');
    let translation = translations[lang];
    
    for (let i = 0; i < keys.length; i++) {
        if (translation && translation[keys[i]]) {
            translation = translation[keys[i]];
        } else {
            // Fallback to English if translation not found
            translation = translations.en;
            for (let j = 0; j < keys.length; j++) {
                if (translation && translation[keys[j]]) {
                    translation = translation[keys[j]];
                } else {
                    return key; // Return key if no translation found
                }
            }
            break;
        }
    }
    
    return translation;
}

// Initialize language on page load
document.addEventListener('DOMContentLoaded', function() {
    // Get saved language preference or default to English
    const savedLang = localStorage.getItem('preferredLanguage') || 'en';
    changeLanguage(savedLang);
    
    // Set up language selector
    const langSelect = document.getElementById('language-select');
    if (langSelect) {
        langSelect.value = savedLang;
        langSelect.addEventListener('change', function() {
            changeLanguage(this.value);
        });
    }
});

// Export for global use
window.changeLanguage = changeLanguage;
window.getTranslation = getTranslation;