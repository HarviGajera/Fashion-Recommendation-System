function showForm(type) {
    document.getElementById('image-form').style.display = (type === 'image') ? 'block' : 'none';
    document.getElementById('text-form').style.display = (type === 'text') ? 'block' : 'none';
}

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
}

document.addEventListener('DOMContentLoaded', function () {
    // Get the file input and relevant elements
    const fileInput = document.getElementById('image-upload');
    const selectedFileDiv = document.getElementById('selected-file');
    const fileName = document.getElementById('file-name-display');
    const removeButton = selectedFileDiv.querySelector('.remove-file');

    // Show selected file name when a file is chosen
    fileInput.addEventListener('change', function () {
        if (this.files && this.files[0]) {
            fileName.textContent = this.files[0].name;
            selectedFileDiv.classList.add('active');
        } else {
            selectedFileDiv.classList.remove('active');
            fileName.textContent = '';
        }
    });

    // Remove file on click
    removeButton.addEventListener('click', function () {
        fileInput.value = '';
        selectedFileDiv.classList.remove('active');
        fileName.textContent = '';
    });
});