import React, { useEffect, useState } from 'react';

export default function WebGazerProvider({ onCalibrationComplete }) {
  const [calibrated, setCalibrated] = useState(false);
  const [clickCount, setClickCount] = useState(0);

  useEffect(() => {
    // Initialize WebGazer
    if (window.webgazer) {
      window.webgazer.clearData();
      window.webgazer
        .setRegression('ridge') /* WebGazer's built in ridge regression for eye tracking */
        .setGazeListener((data, elapsedTime) => {
          // data is an object containing an x and y key which are the x and y prediction coordinates (no bounds limiting)
          if (data == null) {
            return;
          }
          // We can dispatch a custom event for the reading game to listen to
          const gazeEvent = new CustomEvent('gazeData', {
            detail: { x: data.x, y: data.y, time: elapsedTime }
          });
          window.dispatchEvent(gazeEvent);
        })
        .begin()
        .then(() => {
          console.log("WebGazer initialized.");
          // By default webgazer shows video and face tracking overlay
          window.webgazer.showVideoPreview(true).showPredictionPoints(true);
        });
    }

    return () => {
      if (window.webgazer) {
        window.webgazer.pause();
      }
    };
  }, []);

  const handleCalibrationClick = (e) => {
    // A simple 9-point calibration: user clicks on the screen 9 times around edges
    setClickCount(prev => prev + 1);
    
    // Create visual feedback
    const circle = document.createElement("div");
    circle.style.position = "fixed";
    circle.style.left = `${e.clientX - 10}px`;
    circle.style.top = `${e.clientY - 10}px`;
    circle.style.width = "20px";
    circle.style.height = "20px";
    circle.style.borderRadius = "50%";
    circle.style.backgroundColor = "var(--accent-light)";
    circle.style.pointerEvents = "none";
    circle.style.animation = "ping 1s cubic-bezier(0, 0, 0.2, 1) infinite";
    document.body.appendChild(circle);
    
    setTimeout(() => {
      document.body.removeChild(circle);
    }, 1000);

    // After 9 clicks, we consider it calibrated
    if (clickCount >= 8) {
      setCalibrated(true);
      window.webgazer.showVideoPreview(false); // Hide the video to clear up screen space and focus on reading
      onCalibrationComplete();
    }
  };

  if (!calibrated) {
    return (
      <div 
        className="glass-panel calibration-container" 
        onClick={handleCalibrationClick}
        style={{ cursor: 'crosshair', minHeight: '80vh' }}
      >
        <h2>Eye Tracker Calibration</h2>
        <p className="subtitle">
          Please click on 9 different locations across the screen (corners, middle, edges) while looking directly at your mouse cursor.
        </p>
        <div style={{ fontSize: '2rem', fontWeight: 'bold', color: 'var(--accent)' }}>
          {9 - clickCount} clicks left
        </div>
      </div>
    );
  }

  return null;
}
