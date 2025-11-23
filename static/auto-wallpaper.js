// AUTO WALLPAPER LOADER - Detects images in folder
let wallpapers = [];
let current = localStorage.getItem('wallpaper') || 'none';

async function findWallpapers() {
  // Only check bg1.png (the only existing wallpaper)
  try {
    const r = await fetch('/static/wallpapers/bg1.png', { method: 'HEAD' });
    if (r.ok) {
      wallpapers.push({ id: 'bg1', path: '/static/wallpapers/bg1.png' });
    }
  } catch (e) {}
}

function applyWallpaper(id) {
  const body = document.body;
  if (id === 'none') {
    body.style.backgroundImage = '';
  } else {
    const wp = wallpapers.find(w => w.id === id);
    if (wp) {
      body.style.backgroundImage = `url('${wp.path}')`;
      body.style.backgroundSize = 'cover';
      body.style.backgroundPosition = 'center';
      body.style.backgroundAttachment = 'fixed';
      body.style.backgroundRepeat = 'no-repeat';
    }
  }
  localStorage.setItem('wallpaper', id);
  current = id;
  document.querySelectorAll('.wp-btn').forEach(b => {
    b.style.border = b.dataset.wp === id ? '3px solid #00ff00' : '2px solid #666';
  });
}

async function init() {
  await findWallpapers();
  if (wallpapers.length === 0) return;
  
  const div = document.createElement('div');
  div.style.cssText = 'position:fixed;bottom:20px;right:20px;z-index:9999;background:#1a1a1a;border:2px solid #333;border-radius:12px;padding:10px;display:flex;gap:8px;';
  
  const none = document.createElement('button');
  none.className = 'wp-btn';
  none.dataset.wp = 'none';
  none.textContent = '✖';
  none.style.cssText = 'width:50px;height:50px;border-radius:8px;border:2px solid #666;cursor:pointer;background:#333;color:#fff;font-size:20px;';
  none.onclick = () => applyWallpaper('none');
  div.appendChild(none);
  
  wallpapers.forEach(wp => {
    const btn = document.createElement('button');
    btn.className = 'wp-btn';
    btn.dataset.wp = wp.id;
    btn.textContent = wp.id.replace('bg', '');
    btn.style.cssText = `width:50px;height:50px;border-radius:8px;border:2px solid #666;cursor:pointer;background:url('${wp.path}') center/cover;color:#fff;font-weight:bold;text-shadow:0 0 3px #000;`;
    btn.onclick = () => applyWallpaper(wp.id);
    div.appendChild(btn);
  });
  
  document.body.appendChild(div);
  if (current !== 'none') applyWallpaper(current);
  else applyWallpaper('none');
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}
