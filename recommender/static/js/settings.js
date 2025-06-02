// User dropdown toggle
function toggleDropdown() {
    document.getElementById('dropdown-menu').classList.toggle('active');
}

// Close dropdown when clicking outside
window.onclick = function (event) {
    if (!event.target.matches('.user-profile') &&
        !event.target.matches('.profile-picture') &&
        !event.target.matches('.profile-picture-placeholder') &&
        !event.target.matches('.fas.fa-user')) {
        var dropdown = document.getElementById('dropdown-menu');
        if (dropdown && dropdown.classList.contains('active')) {
            dropdown.classList.remove('active');
        }
    }

    // Close modals when clicking outside
    if (event.target.classList.contains('modal')) {
        closeAllModals();
    }
}

// Modal functions
function openUsernameModal() {
    document.getElementById('username-modal').style.display = 'block';
}

function openPasswordModal() {
    document.getElementById('password-modal').style.display = 'block';
}

function openHelpCenter() {
    document.getElementById('help-center-modal').style.display = 'block';
}

function openAbout() {
    document.getElementById('about-modal').style.display = 'block';
}

function closeModal(modalId) {
    document.getElementById(modalId).style.display = 'none';
}

function closeAllModals() {
    const modals = document.getElementsByClassName('modal');
    for (let i = 0; i < modals.length; i++) {
        modals[i].style.display = 'none';
    }
}

// Submit the form inside the modal
function submitForm(modalId) {
    const modal = document.getElementById(modalId);
    const form = modal.querySelector('form');
    form.submit();
}

// Close modal when escape key is pressed
document.addEventListener('keydown', function (event) {
    if (event.key === 'Escape') {
        closeAllModals();
    }
});