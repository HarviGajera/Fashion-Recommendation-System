let originalProfilePicSrc = '';

function toggleDropdown() {
    document.getElementById('dropdown-menu').classList.toggle('active');
}

function openTab(tabName) {
    // Hide all tab content
    var tabContent = document.getElementsByClassName('tab-content');
    for (var i = 0; i < tabContent.length; i++) {
        tabContent[i].classList.remove('active');
    }

    // Remove active class from all tabs
    var tabs = document.getElementsByClassName('tab-btn');
    for (var i = 0; i < tabs.length; i++) {
        tabs[i].classList.remove('active');
    }

    // Show the selected tab content
    document.getElementById(tabName).classList.add('active');

    // Add active class to the clicked tab
    event.currentTarget.classList.add('active');
}

// Edit Profile Modal functions
function openEditProfileModal() {
    document.getElementById('editProfileModal').style.display = 'block';
}

function closeEditProfileModal() {
    const modal = document.getElementById('editProfileModal');
    const preview = document.getElementById('profile-pic-preview');
    const icon = document.getElementById('profile-pic-icon');

    modal.style.display = 'none';

    // Restore original image
    if (originalProfilePicSrc && preview) {
        preview.src = originalProfilePicSrc;
        preview.style.display = 'block';
        if (icon) icon.style.display = 'none';
    }

    // Clear file input
    document.getElementById('profile_picture').value = '';
}

// Close modal when clicking outside of it and handle dropdown
window.onclick = function (event) {
    // Handle modal close
    var modal = document.getElementById('editProfileModal');
    if (event.target == modal) {
        closeEditProfileModal();
    }

    // Handle dropdown close when clicking outside
    if (!event.target.closest('.user-icon')) {
        var dropdown = document.getElementById('dropdown-menu');
        if (dropdown.classList.contains('active')) {
            dropdown.classList.remove('active');
        }
    }
}

// Preview profile image before upload
function previewImage(input) {
    if (input.files && input.files[0]) {
        var reader = new FileReader();

        reader.onload = function (e) {
            var preview = document.getElementById('profile-pic-preview');
            var icon = document.getElementById('profile-pic-icon');

            if (icon) {
                icon.style.display = 'none';
            }

            preview.src = e.target.result;
            preview.style.display = 'block';
        }

        reader.readAsDataURL(input.files[0]);
    }
}

// Auto-hide success messages and store original image on load
document.addEventListener('DOMContentLoaded', function () {
    // Save original profile picture on initial load
    const preview = document.getElementById('profile-pic-preview');
    if (preview && preview.src) {
        originalProfilePicSrc = preview.src;
    }

    // Hide success messages
    setTimeout(function () {
        var successMessages = document.getElementsByClassName('success-message');
        for (var i = 0; i < successMessages.length; i++) {
            successMessages[i].style.display = 'none';
        }
    }, 3000);

    // Auto-open tab based on hash in URL
    const hash = window.location.hash;
    if (hash === '#history') {
        openTab('history');
    } else if (hash === '#settings') {
        openTab('settings');
    } else {
        openTab('favorites'); // default
    }
});

/* Remove from Favourites */
document.addEventListener('DOMContentLoaded', function () {
    const favoriteButtons = document.querySelectorAll('.add-to-favorites');

    favoriteButtons.forEach(button => {
        button.addEventListener('click', function () {
            const productId = this.getAttribute('data-product-id');
            const csrfToken = document.querySelector('meta[name="csrf-token"]').getAttribute('content');

            fetch('/toggle-favorite/', {
                method: 'POST',
                headers: {
                    'X-CSRFToken': csrfToken,
                    'X-Requested-With': 'XMLHttpRequest',
                    'Content-Type': 'application/x-www-form-urlencoded'
                },
                body: `product_id=${productId}`
            })
                .then(response => response.json())
                .then(data => {
                    if (data.status === 'success' && data.action === 'removed') {
                        const productItem = document.querySelector(`.product-item[data-product-id="${productId}"]`);
                        if (productItem) {
                            productItem.remove();
                        }

                        // Show "empty state" if no favorites left
                        const remainingItems = document.querySelectorAll('.product-item');
                        if (remainingItems.length === 0) {
                            const favoritesSection = document.getElementById('favorites');
                            favoritesSection.innerHTML = `
                            <div class="empty-state">
                                <i class="fas fa-heart-broken"></i>
                                <p>You don't have any favorite items yet.</p>
                                <a href="/" class="cta-button">Discover Fashion</a>
                            </div>
                        `;
                        }
                    }
                })
                .catch(error => console.error('Error:', error));
        });
    });
});

function toggleSearchType(type) {
    document.querySelectorAll('.search-tab-btn').forEach(btn => btn.classList.remove('active'));
    document.querySelectorAll('.search-type-content').forEach(tab => tab.classList.remove('active'));

    if (type === 'text') {
        document.querySelector('button[onclick="toggleSearchType(\'text\')"]').classList.add('active');
        document.getElementById('text-searches').classList.add('active');
    } else {
        document.querySelector('button[onclick="toggleSearchType(\'image\')"]').classList.add('active');
        document.getElementById('image-searches').classList.add('active');
    }
}
