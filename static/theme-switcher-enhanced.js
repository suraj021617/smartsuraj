// ENHANCED THEME SWITCHER with 6 Themes
const themes = ['dark', 'light', 'neon', 'ocean', 'sunset', 'forest'];
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
  
  const toggle = document.createElement('div');
  toggle.className = 'theme-toggle';
  toggle.innerHTML = `
    <button class="theme-btn" data-theme="dark" title="Dark">🌙</button>
    <button class="theme-btn" data-theme="light" title="Light">☀️</button>
    <button class="theme-btn" data-theme="neon" title="Neon">⚡</button>
    <button class="theme-btn" data-theme="ocean" title="Ocean">🌊</button>
    <button class="theme-btn" data-theme="sunset" title="Sunset">🌅</button>
    <button class="theme-btn" data-theme="forest" title="Forest">🌲</button>
  `;
  document.body.appendChild(toggle);
  
  toggle.querySelectorAll('.theme-btn').forEach(btn => {
    btn.addEventListener('click', () => setTheme(btn.dataset.theme));
  });
  
  updateButtons();
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initTheme);
} else {
  initTheme();
}

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
