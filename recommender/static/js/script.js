// Function to show a temporary notification
function showNotification(message) {
    // Check if a notification already exists
    let notification = document.querySelector('.notification');

    if (!notification) {
        // Create notification element
        notification = document.createElement('div');
        notification.className = 'notification';
        document.body.appendChild(notification);

        // Style the notification
        notification.style.position = 'fixed';
        notification.style.bottom = '20px';
        notification.style.left = '50%';
        notification.style.transform = 'translateX(-50%)';
        notification.style.backgroundColor = 'rgba(0, 0, 0, 0.7)';
        notification.style.color = 'white';
        notification.style.padding = '10px 20px';
        notification.style.borderRadius = '5px';
        notification.style.zIndex = '1000';
        notification.style.transition = 'opacity 0.3s ease';
    }

    // Set message and show
    notification.textContent = message;
    notification.style.opacity = '1';

    // Hide after delay
    setTimeout(() => {
        notification.style.opacity = '0';

        // Remove from DOM after fade out
        setTimeout(() => {
            if (notification.parentNode) {
                notification.parentNode.removeChild(notification);
            }
        }, 300);
    }, 2000);
}

// Toggle dropdown menu
function toggleDropdown() {
    document.getElementById('dropdown-menu').classList.toggle('active');
}

// Close dropdown when clicking outside
document.addEventListener('click', function (event) {
    // If click is not inside the dropdown or the user icon
    if (!event.target.closest('.user-icon') && !event.target.closest('#dropdown-menu')) {
        const dropdown = document.getElementById('dropdown-menu');
        if (dropdown && dropdown.classList.contains('active')) {
            dropdown.classList.remove('active');
        }
    }
});

// Apply theme on page load
document.addEventListener('DOMContentLoaded', function () {
    const toggle = document.getElementById('dark-mode-toggle');
    const savedTheme = localStorage.getItem('theme');

    if (savedTheme === 'dark') {
        document.body.classList.add('dark-mode');
        toggle.checked = true;
    }

    // Handle toggle interaction
    toggle.addEventListener('change', function () {
        if (this.checked) {
            document.body.classList.add('dark-mode');
            localStorage.setItem('theme', 'dark');
        } else {
            document.body.classList.remove('dark-mode');
            localStorage.setItem('theme', 'light');
        }
    });
});
