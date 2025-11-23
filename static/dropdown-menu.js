document.addEventListener('DOMContentLoaded', function() {
  const dropdowns = document.querySelectorAll('.dropdown-btn');
  
  dropdowns.forEach(btn => {
    btn.addEventListener('click', function() {
      const dropdown = this.parentElement;
      const wasActive = dropdown.classList.contains('active');
      
      document.querySelectorAll('.dropdown').forEach(d => d.classList.remove('active'));
      
      if (!wasActive) {
        dropdown.classList.add('active');
      }
    });
  });
  
  document.addEventListener('click', function(e) {
    if (!e.target.closest('.dropdown')) {
      document.querySelectorAll('.dropdown').forEach(d => d.classList.remove('active'));
    }
  });
});
