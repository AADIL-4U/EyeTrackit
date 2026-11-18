import React, { useState } from 'react';
import WebGazerProvider from './WebGazerProvider';
import ReadingGame from './ReadingGame';

const SAMPLE_TEXT = `The quick brown fox jumps over the lazy dog. Machine learning models can predict how difficult a word is to read based on eye tracking metrics. In this application, your webcam acts as an eye tracker, observing your fixations and saccades in real time. We map your gaze to the screen, extract key linguistic features, and predict the cognitive load for each word.

Reading is a complex cognitive process that requires the coordination of visual perception, attention, and memory. As your eyes move across the text, they do not sweep smoothly but rather jump from word to word in movements known as saccades. During the pauses between these jumps—called fixations—your brain processes the visual information to extract meaning.

By analyzing these patterns, we can identify words that require more effort to understand. Longer fixations or frequent regressions (looking back at previously read text) often indicate difficulties in comprehension or an encounter with unfamiliar vocabulary.`;

function App() {
  const [calibrated, setCalibrated] = useState(false);
  const [textInput, setTextInput] = useState(SAMPLE_TEXT);
  const [playing, setPlaying] = useState(false);

  return (
    <div className="app-container">
      <header>
        <h1>EyeTrack ML</h1>
        <p className="subtitle">Real-time Reading Difficulty Prediction using Webcam</p>
      </header>

      {!calibrated ? (
        <WebGazerProvider onCalibrationComplete={() => setCalibrated(true)} />
      ) : !playing ? (
        <div className="glass-panel" style={{ display: 'flex', flexDirection: 'column', gap: '1rem', width: '100%', maxWidth: '800px' }}>
          <h2>Calibration Complete!</h2>
          <p>Your camera is now tracking your eyes.</p>
          <label style={{ color: 'var(--text-muted)' }}>Enter text to read:</label>
          <textarea 
            value={textInput}
            onChange={(e) => setTextInput(e.target.value)}
            style={{ 
              width: '100%', 
              height: '150px', 
              background: 'rgba(0,0,0,0.3)', 
              color: 'white', 
              border: '1px solid var(--glass-border)',
              padding: '1rem',
              borderRadius: '8px',
              fontFamily: 'inherit',
              resize: 'vertical'
            }}
          />
          <button className="btn" onClick={() => setPlaying(true)} style={{ alignSelf: 'flex-start' }}>
            Start Reading
          </button>
        </div>
      ) : (
        <ReadingGame text={textInput} />
      )}
    </div>
  );
}

export default App;
