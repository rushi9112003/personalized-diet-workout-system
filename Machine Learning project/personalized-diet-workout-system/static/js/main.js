/**
 * main.js - Client-Side Validation and UI Interactions
 */

function validateForm() {
    const ageInput = document.getElementById('age');
    const heightInput = document.getElementById('height');
    const weightInput = document.getElementById('weight');
    const submitBtn = document.getElementById('submitBtn');
    const btnSpinner = document.getElementById('btnSpinner');
    const btnText = document.getElementById('btnText');

    if (ageInput) {
        const age = parseInt(ageInput.value, 10);
        if (isNaN(age) || age < 18 || age > 100) {
            alert('Age must be between 18 and 100 years.');
            ageInput.focus();
            return false;
        }
    }

    if (heightInput) {
        const height = parseFloat(heightInput.value);
        if (isNaN(height) || height < 100 || height > 250) {
            alert('Height must be between 100 cm and 250 cm.');
            heightInput.focus();
            return false;
        }
    }

    if (weightInput) {
        const weight = parseFloat(weightInput.value);
        if (isNaN(weight) || weight < 30 || weight > 250) {
            alert('Weight must be between 30 kg and 250 kg.');
            weightInput.focus();
            return false;
        }
    }

    // Show loading spinner
    if (submitBtn && btnSpinner && btnText) {
        btnSpinner.classList.remove('d-none');
        btnText.textContent = 'Calculating & Ranking via ML...';
        submitBtn.disabled = true;
        // submit form programmatically
        document.getElementById('recommendationForm').submit();
    }

    return true;
}
