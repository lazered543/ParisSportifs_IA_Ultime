const CACHE_NAME="paris-ia-mobile-v1";
const ASSETS=["./","./index.html","./style.css","./app.js","./manifest.json","./icon.svg"];
self.addEventListener("install",e=>{e.waitUntil(caches.open(CACHE_NAME).then(c=>c.addAll(ASSETS)));});
self.addEventListener("fetch",e=>{const url=new URL(e.request.url);if(url.pathname.includes("/data/")||url.pathname.endsWith("tracking_results.csv")){e.respondWith(fetch(e.request).catch(()=>caches.match(e.request)));return;}e.respondWith(caches.match(e.request).then(cached=>cached||fetch(e.request)));});
