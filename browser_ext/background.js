// Function to send URL to the server
async function sendURLToServer() {
    try {
        const currentURL = window.location.href;
        const response = await fetch('http://localhost:5000/add_to_logs', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ url: currentURL })
      });
      console.log('URL sent to server:', currentURL);
    } catch (error) {
      console.error('Error sending URL to server:', error);
    }
}

// Function to send URL periodically
function sendURLPeriodically() {
  sendURLToServer();
  setInterval(sendURLToServer, 5 * 60 * 1000); // 5 minutes in milliseconds
}
  
// Start sending URL periodically when extension is loaded
sendURLPeriodically();