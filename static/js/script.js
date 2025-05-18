// Circular Progress Bar using Chart.js
const ctx = document.getElementById('progressBar').getContext('2d');

// Check if the context is available before proceeding
if (ctx) {
  const progressBar = new Chart(ctx, {
    type: 'doughnut',
    data: {
      datasets: [{
        data: [86, 14], // This will represent the percentage (86% filled)
        backgroundColor: ['#38a169', '#ddd'], // Green for the filled portion, light gray for the rest
        borderWidth: 0, // Remove border width
        hoverOffset: 32, // Slight offset when hovered
      }]
    },
    options: {
      circumference: Math.PI * 57, // Half circle (180 degrees)
      rotation: -Math.PI / 0.035, // Rotate to start from the left (9 o'clock position)
      cutout: '70%', // Creates the doughnut hole (inner empty area)
      responsive: true,
      plugins: {
        tooltip: { enabled: false }, // Disable tooltip for simplicity
        legend: { display: false }, // Disable legend
      },
    }
  });
} else {
  console.error("Failed to get context for the canvas");
}
