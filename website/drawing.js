document.addEventListener("DOMContentLoaded", function () {
  const canvas = document.getElementById("drawing");
  const context = canvas.getContext("2d");

  context.fillStyle = "#fff";
  context.fillRect(0, 0, canvas.width, canvas.height);

  let isDrawing = false;

  function startDrawing(e) {
    isDrawing = true;
    draw(e);
  }

  function stopDrawing() {
    isDrawing = false;
    context.beginPath();
  }

  function draw(e) {
    if (!isDrawing) return;

    context.lineWidth = 5;
    context.lineCap = "round";
    context.strokeStyle = "#000";

    context.lineTo(e.clientX - canvas.offsetLeft, e.clientY - canvas.offsetTop);
    context.stroke();
    context.beginPath();
    context.moveTo(e.clientX - canvas.offsetLeft, e.clientY - canvas.offsetTop);
  }

  canvas.addEventListener("mousedown", startDrawing);
  canvas.addEventListener("mouseup", stopDrawing);
  canvas.addEventListener("mousemove", draw);
});

function saveDrawing() {
  const canvas = document.getElementById("drawing");
  const imgData = canvas.toDataURL("image/png");

  const link = document.createElement("a");
  link.href = imgData;
  link.download = "drawing.png";
  link.click();
}

function clearDrawing() {
  const canvas = document.getElementById("drawing");
  const context = canvas.getContext("2d");
  context.clearRect(0, 0, canvas.width, canvas.height);

  context.fillStyle = "#fff";
  context.fillRect(0, 0, canvas.width, canvas.height);
}
async function sendDrawingForPrediction(predictionUrl) {
  canvas.toBlob(async (blob) => {
    const formData = new FormData();
    formData.append("file", blob, "drawing.png");

    try {
      const response = await fetch(predictionUrl, {
        method: "POST",
        body: formData,
      });
      const result = await response.json();
      alert(JSON.stringify(result));
    } catch (error) {
      console.error("Error:", error);
      alert("An error occurred while trying to predict.");
    }
  });
}

// Event listener for the Predict button
document.getElementById("predict-button").addEventListener("click", () => {
  const predictionType = document.getElementById(
    "prediction-type-select"
  ).value;
  if (predictionType) {
    const predictionUrl = `http://localhost:8000/${predictionType}/`;
    sendDrawingForPrediction(predictionUrl);
  } else {
    alert("Please select a prediction type.");
  }
});
