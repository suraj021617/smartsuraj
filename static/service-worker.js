const CACHE_NAME = 'smartsuraj-v1';
const urlsToCache = [
  '/',
  '/dashboard',
  '/static/carousel.css',
  '/static/countdown.css',
  '/static/my_tracker.css'
];

self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then(cache => cache.addAll(urlsToCache))
  );
});

self.addEventListener('fetch', event => {
  event.respondWith(
    caches.match(event.request)
      .then(response => response || fetch(event.request))
  );
});
