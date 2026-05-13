const CACHE = 'raqaba-v1';
const STATIC = ['/', '/dashboard', '/login', '/static/manifest.json'];

self.addEventListener('install', e => {
  e.waitUntil(caches.open(CACHE).then(c => c.addAll(STATIC)));
});

self.addEventListener('fetch', e => {
  if (e.request.url.includes('/video_feed') || 
      e.request.url.includes('/api/')) return;
  e.respondWith(
    fetch(e.request).catch(() => caches.match(e.request))
  );
});
