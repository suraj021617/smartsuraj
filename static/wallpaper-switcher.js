// WALLPAPER SWITCHER
let currentWallpaper = localStorage.getItem('wallpaper') || 'none';

function setWallpaper(wallpaper) {
  document.documentElement.setAttribute('data-wallpaper', wallpaper);
  localStorage.setItem('wallpaper', wallpaper);
  currentWallpaper = wallpaper;
  updateWallpaperButtons();
}

function updateWallpaperButtons() {
  document.querySelectorAll('.wallpaper-btn').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.wallpaper === currentWallpaper);
  });
}

function initWallpaper() {
  setWallpaper(currentWallpaper);
  
  const selector = document.createElement('div');
  selector.className = 'wallpaper-selector';
  selector.innerHTML = `
    <button class="wallpaper-btn" data-wallpaper="none" title="No Wallpaper"></button>
    <button class="wallpaper-btn" data-wallpaper="custom1" title="Wallpaper 1"></button>
    <button class="wallpaper-btn" data-wallpaper="custom2" title="Wallpaper 2"></button>
    <button class="wallpaper-btn" data-wallpaper="custom3" title="Wallpaper 3"></button>
    <button class="wallpaper-btn" data-wallpaper="custom4" title="Wallpaper 4"></button>
    <button class="wallpaper-btn" data-wallpaper="custom5" title="Wallpaper 5"></button>
  `;
  document.body.appendChild(selector);
  
  selector.querySelectorAll('.wallpaper-btn').forEach(btn => {
    btn.addEventListener('click', () => setWallpaper(btn.dataset.wallpaper));
  });
  
  updateWallpaperButtons();
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initWallpaper);
} else {
  initWallpaper();
}
