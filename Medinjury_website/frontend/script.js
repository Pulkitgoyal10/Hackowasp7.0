// script.js
function uploadImage() {
  const fileInput = document.getElementById('imageInput');
  const resultBox = document.getElementById('resultBox');
  const resultText = document.getElementById('resultText');

  if (!fileInput.files[0]) {
    alert("Please select an image before submitting.");
    return;
  }

  const formData = new FormData();
  formData.append('file', fileInput.files[0]);

  resultText.textContent = "Analyzing...";
  resultBox.classList.remove('hidden');

  fetch('http://127.0.0.1:5000/predict', {
    method: 'POST',
    body: formData
  })
  .then(response => {
    if (!response.ok) {
      throw new Error("Network response was not ok");
    }
    return response.json();
  })
  .then(data => {
    resultText.textContent = `AI Prediction: Class ${data.prediction}`;
  })
  .catch(error => {
    resultText.textContent = `Error: ${error.message}`;
  });
} 
