// Mobile Navigation Toggle
document.addEventListener('DOMContentLoaded', function() {
    // Smooth scrolling for navigation links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                window.scrollTo({
                    top: target.offsetTop - 80,
                    behavior: 'smooth'
                });
            }
        });
    });

    // Highlight active navigation item on scroll
    window.addEventListener('scroll', function() {
        const sections = document.querySelectorAll('section');
        const navLinks = document.querySelectorAll('.nav-menu a');
        
        let current = '';
        
        sections.forEach(section => {
            const sectionTop = section.offsetTop;
            const sectionHeight = section.clientHeight;
            
            if (pageYOffset >= (sectionTop - 100)) {
                current = section.getAttribute('id');
            }
        });
        
        navLinks.forEach(link => {
            link.classList.remove('active');
            if (link.getAttribute('href') === `#${current}`) {
                link.classList.add('active');
            }
        });
    });

    // Demo placeholder interaction
    const demoPlaceholder = document.querySelector('.demo-placeholder');
    if (demoPlaceholder) {
        demoPlaceholder.addEventListener('click', function() {
            const currentLang = document.documentElement.getAttribute('lang') || 'en';
            const messages = {
                en: 'AIPlatform Quantum Revolution Demo would play here. In a full implementation, this would launch an interactive demonstration of the quantum-AI platform.',
                ru: 'Демонстрация квантовой революции AIPlatform будет воспроизводиться здесь. В полной реализации это запустит интерактивную демонстрацию квантово-ИИ платформы.',
                zh: 'AIPlatform量子革命演示将在此播放。在完整实现中，这将启动量子AI平台的交互式演示。',
                ar: 'عرض ثورة الذكاء الاصطناعي الكمي AIPlatform سيتم هنا. في التنفيذ الكامل، سيطلق هذا عرضًا تفاعليًا لمنصة الذكاء الاصطناعي الكمي.'
            };
            alert(messages[currentLang] || messages.en);
        });
    }

    // Code preview animation
    const codePreview = document.querySelector('.code-preview');
    if (codePreview) {
        // Add typing animation effect
        const codeLines = codePreview.querySelectorAll('pre code');
        codeLines.forEach(line => {
            line.style.opacity = '0';
            line.style.transform = 'translateY(20px)';
        });
        
        setTimeout(() => {
            codeLines.forEach((line, index) => {
                setTimeout(() => {
                    line.style.transition = 'all 0.5s ease';
                    line.style.opacity = '1';
                    line.style.transform = 'translateY(0)';
                }, index * 200);
            });
        }, 500);
    }

    // Feature card hover effects
    const featureCards = document.querySelectorAll('.feature-card');
    featureCards.forEach(card => {
        card.addEventListener('mouseenter', function() {
            this.style.transform = 'translateY(-10px)';
        });
        
        card.addEventListener('mouseleave', function() {
            this.style.transform = 'translateY(0)';
        });
    });

    // Initialize animations on scroll
    const observerOptions = {
        root: null,
        rootMargin: '0px',
        threshold: 0.1
    };

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('animate-in');
            }
        });
    }, observerOptions);

    // Observe elements for animation
    document.querySelectorAll('.feature-card, .doc-card, .example-card, .community-card').forEach(card => {
        observer.observe(card);
    });

    // Form validation for any future forms
    const forms = document.querySelectorAll('form');
    forms.forEach(form => {
        form.addEventListener('submit', function(e) {
            e.preventDefault();
            
            // Simple form validation
            let isValid = true;
            const inputs = this.querySelectorAll('input[required]');
            
            inputs.forEach(input => {
                if (!input.value.trim()) {
                    isValid = false;
                    input.style.borderColor = '#ef4444';
                } else {
                    input.style.borderColor = '';
                }
            });
            
            if (isValid) {
                const currentLang = document.documentElement.getAttribute('lang') || 'en';
                const messages = {
                    en: 'Form submitted successfully! In a full implementation, this would send your message to our team.',
                    ru: 'Форма успешно отправлена! В полной реализации это отправит ваше сообщение нашей команде.',
                    zh: '表单提交成功！在完整实现中，这会将您的消息发送给我们的团队。',
                    ar: 'تم إرسال النموذج بنجاح! في التنفيذ الكامل، سيؤدي هذا إلى إرسال رسالتك إلى فريقنا.'
                };
                alert(messages[currentLang] || messages.en);
                form.reset();
            }
        });
    });

    // Theme toggle (if implemented)
    const themeToggle = document.getElementById('theme-toggle');
    if (themeToggle) {
        themeToggle.addEventListener('click', function() {
            document.body.classList.toggle('dark-theme');
            localStorage.setItem('theme', document.body.classList.contains('dark-theme') ? 'dark' : 'light');
        });
    }

    // Initialize theme from localStorage
    if (localStorage.getItem('theme') === 'dark') {
        document.body.classList.add('dark-theme');
    }
});

// Quantum particle background effect (simplified)
function createQuantumParticles() {
    const hero = document.querySelector('.hero');
    if (!hero) return;
    
    // Create a few floating particles for visual effect
    for (let i = 0; i < 20; i++) {
        const particle = document.createElement('div');
        particle.className = 'quantum-particle';
        particle.style.left = Math.random() * 100 + '%';
        particle.style.top = Math.random() * 100 + '%';
        particle.style.animationDelay = Math.random() * 5 + 's';
        particle.style.opacity = Math.random() * 0.5 + 0.1;
        particle.innerHTML = '⚛️';
        
        // Add to hero section
        hero.appendChild(particle);
    }
}

// Add quantum particles on load
document.addEventListener('DOMContentLoaded', createQuantumParticles);

// Performance monitoring
function trackPerformance() {
    if ('performance' in window) {
        window.addEventListener('load', function() {
            setTimeout(function() {
                const perfData = performance.getEntriesByType('navigation')[0];
                console.log('Page Load Time:', perfData.loadEventEnd - perfData.fetchStart, 'ms');
            }, 0);
        });
    }
}

trackPerformance();

// Error handling
window.addEventListener('error', function(e) {
    console.error('Global error:', e.error);
    // In production, you might want to send this to an error tracking service
});

// Add animation classes for CSS
document.addEventListener('DOMContentLoaded', function() {
    // Add animation classes to elements
    const animatedElements = document.querySelectorAll('.feature-card, .doc-card, .example-card, .community-card');
    animatedElements.forEach((el, index) => {
        el.style.transitionDelay = (index * 0.1) + 's';
    });
});

// Language change handler
function changeLanguage(lang) {
    // This function is now handled by i18n.js
    // We keep it here for backward compatibility
    if (window.changeLanguage) {
        window.changeLanguage(lang);
    }
}