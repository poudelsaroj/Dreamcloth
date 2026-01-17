/* =====================================================
   DreamCloth - Academic Project Page Scripts
   ===================================================== */

// DOM Ready
document.addEventListener('DOMContentLoaded', function() {
    initScrollEffects();
    initNavbarScroll();
    initIntersectionObserver();
});

// Scroll to Top Functionality
function scrollToTop() {
    window.scrollTo({
        top: 0,
        behavior: 'smooth'
    });
}

// Initialize Scroll Effects
function initScrollEffects() {
    const scrollTopBtn = document.querySelector('.scroll-top');
    
    if (scrollTopBtn) {
        window.addEventListener('scroll', function() {
            if (window.pageYOffset > 300) {
                scrollTopBtn.classList.add('visible');
            } else {
                scrollTopBtn.classList.remove('visible');
            }
        });
    }
}

// Navbar Scroll Effect
function initNavbarScroll() {
    const navbar = document.querySelector('.navbar');
    let lastScroll = 0;
    
    window.addEventListener('scroll', function() {
        const currentScroll = window.pageYOffset;
        
        if (currentScroll > 100) {
            navbar.style.boxShadow = '0 2px 10px rgba(0,0,0,0.1)';
        } else {
            navbar.style.boxShadow = 'none';
        }
        
        lastScroll = currentScroll;
    });
}

// Copy BibTeX to Clipboard
function copyBibtex() {
    const bibtexCode = document.querySelector('.bibtex-container code');
    const copyBtn = document.querySelector('.copy-btn');
    
    if (bibtexCode) {
        const text = bibtexCode.textContent;
        
        navigator.clipboard.writeText(text).then(function() {
            // Visual feedback
            const originalText = copyBtn.innerHTML;
            copyBtn.innerHTML = '<i class="fas fa-check"></i> Copied!';
            copyBtn.classList.add('copied');
            
            setTimeout(function() {
                copyBtn.innerHTML = originalText;
                copyBtn.classList.remove('copied');
            }, 2000);
        }).catch(function(err) {
            console.error('Failed to copy: ', err);
            // Fallback for older browsers
            fallbackCopy(text);
        });
    }
}

// Fallback Copy Method
function fallbackCopy(text) {
    const textArea = document.createElement('textarea');
    textArea.value = text;
    textArea.style.position = 'fixed';
    textArea.style.left = '-999999px';
    document.body.appendChild(textArea);
    textArea.focus();
    textArea.select();
    
    try {
        document.execCommand('copy');
        const copyBtn = document.querySelector('.copy-btn');
        copyBtn.innerHTML = '<i class="fas fa-check"></i> Copied!';
        copyBtn.classList.add('copied');
        
        setTimeout(function() {
            copyBtn.innerHTML = '<i class="fas fa-copy"></i> Copy';
            copyBtn.classList.remove('copied');
        }, 2000);
    } catch (err) {
        console.error('Fallback copy failed: ', err);
    }
    
    document.body.removeChild(textArea);
}

// Intersection Observer for Animations
function initIntersectionObserver() {
    const observerOptions = {
        root: null,
        rootMargin: '0px',
        threshold: 0.1
    };
    
    const observer = new IntersectionObserver(function(entries) {
        entries.forEach(function(entry) {
            if (entry.isIntersecting) {
                entry.target.classList.add('fade-in-up');
                observer.unobserve(entry.target);
            }
        });
    }, observerOptions);
    
    // Observe sections
    const sections = document.querySelectorAll('.section');
    sections.forEach(function(section) {
        section.style.opacity = '0';
        observer.observe(section);
    });
}

// Smooth Scroll for Anchor Links
document.querySelectorAll('a[href^="#"]').forEach(function(anchor) {
    anchor.addEventListener('click', function(e) {
        e.preventDefault();
        const targetId = this.getAttribute('href');
        
        if (targetId === '#') {
            scrollToTop();
            return;
        }
        
        const targetElement = document.querySelector(targetId);
        if (targetElement) {
            const navbarHeight = document.querySelector('.navbar').offsetHeight;
            const targetPosition = targetElement.getBoundingClientRect().top + window.pageYOffset - navbarHeight - 20;
            
            window.scrollTo({
                top: targetPosition,
                behavior: 'smooth'
            });
        }
    });
});

// Video Placeholder Click Handler
document.querySelectorAll('.video-placeholder').forEach(function(placeholder) {
    placeholder.style.cursor = 'pointer';
    placeholder.addEventListener('click', function() {
        // This will be replaced with actual video loading when videos are ready
        console.log('Video placeholder clicked - video coming soon!');
    });
});

// Lazy Loading for Videos (when they're added)
function initLazyVideos() {
    const lazyVideos = document.querySelectorAll('video[data-src]');
    
    if ('IntersectionObserver' in window) {
        const videoObserver = new IntersectionObserver(function(entries) {
            entries.forEach(function(entry) {
                if (entry.isIntersecting) {
                    const video = entry.target;
                    video.src = video.dataset.src;
                    video.load();
                    videoObserver.unobserve(video);
                }
            });
        });
        
        lazyVideos.forEach(function(video) {
            videoObserver.observe(video);
        });
    }
}

// Handle Responsive Navigation (for future mobile menu)
function toggleMobileMenu() {
    const navLinks = document.querySelector('.nav-links');
    if (navLinks) {
        navLinks.classList.toggle('active');
    }
}

// Preload critical resources
function preloadResources() {
    // Add critical font preloading if needed
    const fonts = [
        'https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,600;8..60,700&display=swap'
    ];
    
    fonts.forEach(function(font) {
        const link = document.createElement('link');
        link.rel = 'preload';
        link.as = 'style';
        link.href = font;
        document.head.appendChild(link);
    });
}

// Error handling for images
document.querySelectorAll('img').forEach(function(img) {
    img.addEventListener('error', function() {
        this.style.display = 'none';
        console.warn('Failed to load image:', this.src);
    });
});

// Console Easter Egg
console.log('%c🧵 DreamCloth', 'font-size: 24px; font-weight: bold; color: #1a365d;');
console.log('%cWorld-Prior Distillation for Physically Plausible Simulation Parameters', 'font-size: 12px; color: #718096;');
console.log('%cTribhuvan University - SIGGRAPH 2026 Submission', 'font-size: 10px; color: #a0aec0;');