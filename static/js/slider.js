// 3D Slider Dashboard Controller
class SliderDashboard {
    constructor() {
        this.currentMode = 'predictions';
        this.sliders = {
            predictions: { active: 3, items: [] },
            history: { active: 3, items: [] },
            methods: { active: 3, items: [] },
            learning: { active: 3, items: [] }
        };
        this.init();
    }

    init() {
        this.setupTabs();
        this.setupSliders();
        this.setupNavigation();
        this.setupTouchEvents();
        this.loadShow('predictions');
    }

    setupTabs() {
        const tabs = document.querySelectorAll('.tab-btn');
        tabs.forEach(tab => {
            tab.addEventListener('click', (e) => {
                const mode = e.target.dataset.mode;
                this.switchMode(mode);
            });
        });
    }

    setupSliders() {
        Object.keys(this.sliders).forEach(mode => {
            const container = document.getElementById(`${mode}-slider`);
            if (container) {
                const items = container.querySelectorAll('.item');
                this.sliders[mode].items = Array.from(items);
                
                // Ensure we have enough items and set proper active index
                if (items.length > 0) {
                    this.sliders[mode].active = Math.min(3, Math.floor(items.length / 2));
                }
            }
        });
    }

    setupNavigation() {
        // Setup navigation for each mode
        Object.keys(this.sliders).forEach(mode => {
            const nextBtn = document.getElementById(`${mode.substring(0, 4)}-next`);
            const prevBtn = document.getElementById(`${mode.substring(0, 4)}-prev`);
            
            if (nextBtn) {
                nextBtn.addEventListener('click', () => this.next(mode));
            }
            if (prevBtn) {
                prevBtn.addEventListener('click', () => this.prev(mode));
            }
        });

        // Keyboard navigation
        document.addEventListener('keydown', (e) => {
            if (e.key === 'ArrowLeft') this.prev(this.currentMode);
            if (e.key === 'ArrowRight') this.next(this.currentMode);
        });
    }

    setupTouchEvents() {
        let startX = 0;
        let startY = 0;

        document.addEventListener('touchstart', (e) => {
            startX = e.touches[0].clientX;
            startY = e.touches[0].clientY;
        });

        document.addEventListener('touchend', (e) => {
            if (!startX || !startY) return;

            const endX = e.changedTouches[0].clientX;
            const endY = e.changedTouches[0].clientY;
            
            const diffX = startX - endX;
            const diffY = startY - endY;

            // Only handle horizontal swipes
            if (Math.abs(diffX) > Math.abs(diffY) && Math.abs(diffX) > 50) {
                if (diffX > 0) {
                    this.next(this.currentMode);
                } else {
                    this.prev(this.currentMode);
                }
            }

            startX = 0;
            startY = 0;
        });
    }

    switchMode(mode) {
        // Update tab buttons
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.classList.remove('active');
        });
        document.querySelector(`[data-mode="${mode}"]`).classList.add('active');

        // Hide all sliders
        document.querySelectorAll('.slider-container').forEach(container => {
            container.classList.add('hidden');
        });

        // Show selected slider
        const targetContainer = document.getElementById(`${mode}-slider`);
        if (targetContainer) {
            targetContainer.classList.remove('hidden');
            this.currentMode = mode;
            this.loadShow(mode);
        }
    }

    next(mode) {
        const slider = this.sliders[mode];
        if (!slider || !slider.items.length) return;
        
        slider.active = slider.active + 1 < slider.items.length ? slider.active + 1 : slider.active;
        this.loadShow(mode);
    }

    prev(mode) {
        const slider = this.sliders[mode];
        if (!slider || !slider.items.length) return;
        
        slider.active = slider.active - 1 >= 0 ? slider.active - 1 : slider.active;
        this.loadShow(mode);
    }

    loadShow(mode) {
        const slider = this.sliders[mode];
        if (!slider || !slider.items.length) return;

        const items = slider.items;
        const active = slider.active;

        // Reset all items
        items.forEach(item => {
            item.style.transform = 'none';
            item.style.zIndex = '1';
            item.style.filter = 'none';
            item.style.opacity = '1';
        });

        // Style active item
        if (items[active]) {
            items[active].style.transform = 'none';
            items[active].style.zIndex = '10';
            items[active].style.filter = 'none';
            items[active].style.opacity = '1';
        }

        // Style items to the right
        let stt = 0;
        for (let i = active + 1; i < items.length; i++) {
            stt++;
            const offset = Math.min(stt * 120, 300);
            const scale = Math.max(1 - 0.2 * stt, 0.4);
            const opacity = stt > 2 ? 0 : Math.max(0.6 - 0.1 * stt, 0.3);
            
            items[i].style.transform = `translateX(${offset}px) scale(${scale}) perspective(16px) rotateY(-${Math.min(stt * 2, 10)}deg)`;
            items[i].style.zIndex = `${10 - stt}`;
            items[i].style.filter = `blur(${Math.min(stt * 2, 8)}px)`;
            items[i].style.opacity = opacity;
        }

        // Style items to the left
        stt = 0;
        for (let i = active - 1; i >= 0; i--) {
            stt++;
            const offset = Math.min(stt * 120, 300);
            const scale = Math.max(1 - 0.2 * stt, 0.4);
            const opacity = stt > 2 ? 0 : Math.max(0.6 - 0.1 * stt, 0.3);
            
            items[i].style.transform = `translateX(-${offset}px) scale(${scale}) perspective(16px) rotateY(${Math.min(stt * 2, 10)}deg)`;
            items[i].style.zIndex = `${10 - stt}`;
            items[i].style.filter = `blur(${Math.min(stt * 2, 8)}px)`;
            items[i].style.opacity = opacity;
        }

        // Update progress circles for learning mode
        if (mode === 'learning') {
            this.updateProgressCircles();
        }
    }

    updateProgressCircles() {
        const circles = document.querySelectorAll('.progress-circle');
        circles.forEach(circle => {
            const fillElement = circle.querySelector('.circle-fill');
            if (fillElement) {
                const progress = parseInt(fillElement.dataset.progress) || 0;
                const degrees = (progress / 100) * 360;
                circle.style.background = `conic-gradient(#4299e1 ${degrees}deg, #e2e8f0 ${degrees}deg)`;
            }
        });
    }
}

// Initialize dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    const dashboard = new SliderDashboard();
    
    // Add smooth transitions
    const style = document.createElement('style');
    style.textContent = `
        .item {
            transition: all 0.6s cubic-bezier(0.25, 0.46, 0.45, 0.94) !important;
        }
    `;
    document.head.appendChild(style);
    
    // Add loading animation
    setTimeout(() => {
        document.body.classList.add('loaded');
    }, 100);
});

// Add number animation for methods mode
document.addEventListener('click', (e) => {
    if (e.target.dataset.mode === 'methods') {
        setTimeout(() => {
            const numbers = document.querySelectorAll('.big-number, .mini-number');
            numbers.forEach(num => {
                const finalValue = parseInt(num.textContent);
                if (!isNaN(finalValue)) {
                    let currentValue = 0;
                    const increment = finalValue / 30;
                    const timer = setInterval(() => {
                        currentValue += increment;
                        if (currentValue >= finalValue) {
                            currentValue = finalValue;
                            clearInterval(timer);
                        }
                        num.textContent = Math.floor(currentValue);
                    }, 50);
                }
            });
        }, 300);
    }
});