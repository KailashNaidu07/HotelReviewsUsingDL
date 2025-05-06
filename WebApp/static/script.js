function getPrediction() {
    const reviewText = document.getElementById('review-text').value;
    
    if (!reviewText.trim()) {
        alert('Please enter a review');
        return;
    }

    if (reviewText.trim().length < 100) {
        alert('Please enter a review with at least 100 characters');
        return;
    }

    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ review: reviewText })
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            alert(data.error);
            return;
        }
        
        displayResults(data.rating, data.confidence);
    })
    .catch(error => {
        console.error('Error:', error);
        alert('An error occurred while analyzing the review');
    });
}

function displayResults(rating, confidence) {
    // Show result container
    document.getElementById('result-container').style.display = 'block';
    
    // Display stars
    const starsContainer = document.getElementById('stars-container');
    starsContainer.innerHTML = '';
    for (let i = 1; i <= 5; i++) {
        const star = document.createElement('i');
        star.className = i <= rating ? 'fas fa-star stars' : 'far fa-star stars';
        starsContainer.appendChild(star);
    }
    
    // Update confidence bar
    const confidenceFill = document.getElementById('confidence-fill');
    confidenceFill.style.width = `${confidence * 100}%`;
    
    // Update confidence text
    document.getElementById('confidence-text').textContent = 
        `${(confidence * 100).toFixed(1)}% confident`;
}

function updateCharCount() {
    const reviewText = document.getElementById('review-text').value;
    const charCount = document.getElementById('char-count');
    const length = reviewText.trim().length;
    
    charCount.textContent = `${length} characters`;
    charCount.className = `char-count ${length < 100 ? 'invalid' : ''}`;
} 