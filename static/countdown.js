function getNextDrawDate() {
    const now = new Date();
    const drawDays = [3, 6, 0]; // Wed, Sat, Sun
    const drawHour = 19; // 7 PM
    
    let nextDraw = new Date(now);
    nextDraw.setHours(drawHour, 0, 0, 0);
    
    // If today's draw time passed, move to next day
    if (now.getHours() >= drawHour) {
        nextDraw.setDate(nextDraw.getDate() + 1);
    }
    
    // Find next draw day
    while (!drawDays.includes(nextDraw.getDay())) {
        nextDraw.setDate(nextDraw.getDate() + 1);
    }
    
    return nextDraw;
}

function updateCountdown() {
    const nextDraw = getNextDrawDate();
    const now = new Date();
    const diff = nextDraw - now;
    
    if (diff <= 0) {
        location.reload();
        return;
    }
    
    const days = Math.floor(diff / (1000 * 60 * 60 * 24));
    const hours = Math.floor((diff % (1000 * 60 * 60 * 24)) / (1000 * 60 * 60));
    const minutes = Math.floor((diff % (1000 * 60 * 60)) / (1000 * 60));
    const seconds = Math.floor((diff % (1000 * 60)) / 1000);
    
    document.getElementById('days').textContent = String(days).padStart(2, '0');
    document.getElementById('hours').textContent = String(hours).padStart(2, '0');
    document.getElementById('minutes').textContent = String(minutes).padStart(2, '0');
    document.getElementById('seconds').textContent = String(seconds).padStart(2, '0');
    
    const options = { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' };
    document.getElementById('draw-date').textContent = nextDraw.toLocaleDateString('en-US', options);
}

updateCountdown();
setInterval(updateCountdown, 1000);
