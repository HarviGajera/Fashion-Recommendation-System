document.addEventListener('DOMContentLoaded', function () {
    // Get CSRF token from meta tag
    const csrfToken = document.querySelector('meta[name="csrf-token"]')?.getAttribute('content');

    if (!csrfToken) {
        console.error('CSRF token not found');
        return;
    }

    // Add click event listeners to all favorite buttons
    document.querySelectorAll('.add-to-favorites').forEach(button => {
        button.addEventListener('click', function (e) {
            e.preventDefault();
            e.stopPropagation();

            const productId = this.getAttribute('data-product-id');
            if (!productId) {
                console.error('Product ID not found on button');
                return;
            }

            toggleFavoriteStatus(productId, this);
        });
    });

    // Function to toggle favorite status via AJAX
    function toggleFavoriteStatus(productId, buttonElement) {
        // Create form data
        const formData = new FormData();
        formData.append('product_id', productId);

        // Send AJAX request
        fetch('/toggle-favorite/', {
            method: 'POST',
            headers: {
                'X-Requested-With': 'XMLHttpRequest',
                'X-CSRFToken': csrfToken
            },
            body: formData
        })
            .then(response => {
                if (!response.ok) {
                    throw new Error(`Network response error: ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                console.log('Toggle favorite response:', data);

                if (data.status === 'success') {
                    // Update UI based on response
                    const icon = buttonElement.querySelector('i');

                    if (data.action === 'added') {
                        // Added to favorites
                        icon.classList.remove('far');
                        icon.classList.add('fas');
                        buttonElement.classList.add('active');
                        showNotification('Added to favorites');
                    } else {
                        // Removed from favorites
                        icon.classList.remove('fas');
                        icon.classList.add('far');
                        buttonElement.classList.remove('active');
                        showNotification('Removed from favorites');
                    }
                } else {
                    showNotification('Error: ' + (data.message || 'Failed to update favorites'));
                }
            })
            .catch(error => {
                console.error('Error toggling favorite status:', error);
                showNotification('Error: Could not update favorites');
            });
    }

    // Function to show a temporary notification if not already defined
    // This ensures we have this function available regardless of script load order
    if (typeof showNotification !== 'function') {
        window.showNotification = function (message) {
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
        };
    }
});