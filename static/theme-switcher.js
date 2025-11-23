// THEME SWITCHER - Minimal & Fast
const themes = ['dark', 'light', 'neon'];
let currentTheme = localStorage.getItem('theme') || 'dark';

function setTheme(theme) {
  document.documentElement.setAttribute('data-theme', theme);
  localStorage.setItem('theme', theme);
  currentTheme = theme;
  updateButtons();
}

function updateButtons() {
  document.querySelectorAll('.theme-btn').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.theme === currentTheme);
  });
}

function initTheme() {
  setTheme(currentTheme);
  
  // Add theme toggle to page
  const toggle = document.createElement('div');
  toggle.className = 'theme-toggle';
  toggle.innerHTML = `
    <button class="theme-btn" data-theme="dark" title="Dark Mode">🌙</button>
    <button class="theme-btn" data-theme="light" title="Light Mode">☀️</button>
    <button class="theme-btn" data-theme="neon" title="Neon Mode">⚡</button>
  `;
  document.body.appendChild(toggle);
  
  // Event listeners
  toggle.querySelectorAll('.theme-btn').forEach(btn => {
    btn.addEventListener('click', () => setTheme(btn.dataset.theme));
  });
  
  updateButtons();
}

// Auto-init
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initTheme);
} else {
  initTheme();
}

// Animate elements on scroll
const observer = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      entry.target.classList.add('animate-fade');
    }
  });
}, { threshold: 0.1 });

document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('.card, .btn, table').forEach(el => observer.observe(el));
});
