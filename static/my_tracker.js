function loadTracked() {
    const tracked = JSON.parse(localStorage.getItem('myTracked') || '[]');
    const list = document.getElementById('trackedList');
    
    if (tracked.length === 0) {
        list.innerHTML = '<p style="text-align:center;color:#999;">No tracked numbers yet</p>';
        return;
    }
    
    list.innerHTML = tracked.map((item, idx) => `
        <div class="tracked-item">
            <div>
                <div class="number">${item.number}</div>
                ${item.note ? `<div class="note">${item.note}</div>` : ''}
                <div class="date">Added: ${item.date}</div>
            </div>
            <button onclick="removeNumber(${idx})">🗑️ Remove</button>
        </div>
    `).join('');
}

function addNumber() {
    const num = document.getElementById('numberInput').value;
    const note = document.getElementById('noteInput').value;
    
    if (!/^\d{4}$/.test(num)) {
        alert('Please enter a valid 4-digit number');
        return;
    }
    
    const tracked = JSON.parse(localStorage.getItem('myTracked') || '[]');
    tracked.push({
        number: num,
        note: note,
        date: new Date().toLocaleDateString()
    });
    
    localStorage.setItem('myTracked', JSON.stringify(tracked));
    document.getElementById('numberInput').value = '';
    document.getElementById('noteInput').value = '';
    loadTracked();
}

function removeNumber(idx) {
    const tracked = JSON.parse(localStorage.getItem('myTracked') || '[]');
    tracked.splice(idx, 1);
    localStorage.setItem('myTracked', JSON.stringify(tracked));
    loadTracked();
}

loadTracked();
