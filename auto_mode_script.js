    <script>
        // Toggle between manual and auto prediction modes
        document.getElementById('autoMode').addEventListener('change', function() {
            const textarea = document.querySelector('textarea[name="predictions"]');
            const button = document.querySelector('button[type="submit"]');
            
            if (this.checked) {
                textarea.disabled = true;
                textarea.placeholder = "🚀 AUTO MODE: All predictors will be used automatically";
                button.innerHTML = '🔍 Check ALL Predictor Matches';
            } else {
                textarea.disabled = false;
                textarea.placeholder = "1234, 5678, 9012 (manual input)";
                button.innerHTML = '🔍 Check Matches';
            }
        });
    </script>