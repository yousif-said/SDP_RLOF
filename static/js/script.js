//selecting elements from the DOM
const textareaInput = document.querySelector("textarea"); //input area
const generateButton = document.querySelector(".buttons button"); //button for generating a continuation
const toggleOptions = document.querySelectorAll(".toggle input"); //options as toggle
const predictionBox = document.querySelector(".box textarea[readonly]"); //prediction box
const feedbackButtons = document.querySelectorAll(".feedback-buttons button"); //Feedback buttons

//function for continuation generation
generateButton.addEventListener("click", () => {
  //get input prompt and selected toggle option
  const inputPrompt = textareaInput.value;
  const toxicOption = document.querySelector('input[name="continuation"]:checked').value;

  if (!inputPrompt) {
    alert("Please enter a prompt before generating!");
    return;
  }

  simulateBackendRequest(inputPrompt, toxicOption)
    .then((response) => {
      predictionBox.value = response.prediction;
    })
    .catch((error) => {
      console.error("Error:", error);
      predictionBox.value = "An error occurred. Please try again.";
    });
});

//function to simulate backend processing for now
function simulateBackendRequest(prompt, toxicOption) {
  return new Promise((resolve) => {
    setTimeout(() => {
      const prediction =
        toxicOption === "toxic"
          ? `Toxic response generated for: "${prompt}"`
          : `Nontoxic response generated for: "${prompt}"`;

      resolve({ prediction });
    }, 1000);
  });
}

//feedback button event listeners
feedbackButtons.forEach((button) => {
  button.addEventListener("click", () => {
    const feedback = button.classList.contains("yes") ? "Yes" : "No";
    alert(`You selected: ${feedback}`);
  });
});
