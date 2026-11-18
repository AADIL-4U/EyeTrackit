import React, { useState, useEffect, useRef } from 'react';

export default function ReadingGame({ text }) {
  const [words, setWords] = useState([]);
  const [paragraphs, setParagraphs] = useState([]);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [results, setResults] = useState([]);
  
  // Ref to store eye movement metrics for each word
  // data: { word_id: { first_fixation_ms, first_pass_ms, fixation_count, regression_count, skip_flag, lastGazeTime } }
  const metricsRef = useRef({});
  const wordElementsRef = useRef({});
  const lastFixatedWordIdRef = useRef(-1);

  // Calculate thresholds based on percentiles to ensure visual contrast
  const thresholds = React.useMemo(() => {
    if (!results || results.length === 0) return null;
    const values = results.map(r => r.predicted_difficulty_ms).filter(v => v > 0).sort((a, b) => a - b);
    if (values.length === 0) return null;

    const getPercentile = (p) => {
      const idx = Math.floor(p * (values.length - 1));
      return values[idx];
    };

    return {
      p60: getPercentile(0.60), // Green threshold
      p80: getPercentile(0.80), // Yellow threshold
      p95: getPercentile(0.95), // Orange threshold
    };
  }, [results]);

  const getDifficultyColor = (ms) => {
    if (!thresholds || ms <= 0) return 'inherit';

    if (ms <= thresholds.p60) return '#4ade80';   // Green (Easy: Bottom 60%)
    if (ms <= thresholds.p80) return '#facc15';   // Yellow (Moderate: 60-80%)
    if (ms <= thresholds.p95) return '#fb923c';   // Orange (Hard: 80-95%)
    return '#f87171';                            // Red (Very Hard: Top 5%)
  };

  useEffect(() => {
    // Split text into paragraphs and words, and initialize metrics
    const pGroups = text.split(/\n\s*\n/);
    let currentId = 0;
    
    const parsedParagraphs = pGroups.map((pText) => {
      const pWords = pText.trim().split(/\s+/).filter(Boolean);
      return pWords.map((word) => {
         currentId++;
         return { word, id: currentId };
      });
    });
    setParagraphs(parsedParagraphs);
    
    // We also need the flat list to initialize metrics
    const flatWords = parsedParagraphs.flat();
    setWords(flatWords);
    
    // Initialize metrics
    const initialMetrics = {};
    flatWords.forEach(w => {
      initialMetrics[w.id] = {
        word: w.word,
        word_id: w.id,
        first_fixation_ms: 0,
        first_pass_ms: 0,
        fixation_count: 0,
        regression_count: 0,
        skip_flag: 1, // Assume skipped until gazed
        lastGazeTime: null,
        totalGazeTime: 0
      };
    });
    metricsRef.current = initialMetrics;

    const handleGaze = (e) => {
      if (isAnalyzing) return; // Stop tracking during analysis
      
      const { x, y, time } = e.detail;
      let fixatedWordId = -1;

      // Check which word bounding box the gaze falls into
      for (const [idStr, el] of Object.entries(wordElementsRef.current)) {
        if (!el) continue;
        const rect = el.getBoundingClientRect();
        
        // Add a small padding tolerance to the bounding box
        const padding = 20; 
        if (x >= rect.left - padding && x <= rect.right + padding &&
            y >= rect.top - padding && y <= rect.bottom + padding) {
          fixatedWordId = parseInt(idStr);
          break;
        }
      }

      if (fixatedWordId !== -1) {
        const metric = metricsRef.current[fixatedWordId];
        metric.skip_flag = 0;
        
        const lastWordId = lastFixatedWordIdRef.current;
        
        // If we transitioned to a NEW word
        if (lastWordId !== fixatedWordId) {
          metric.fixation_count += 1;
          
          // Check for regression (went backwards)
          if (lastWordId !== -1 && fixatedWordId < lastWordId) {
            metric.regression_count += 1;
          }
          
          metric.lastGazeTime = time;
        } else {
          // Maintaining gaze on the SAME word
          const diff = time - metric.lastGazeTime;
          metric.totalGazeTime += diff;
          
          // First pass calculation
          if (metric.fixation_count === 1) {
             metric.first_pass_ms += diff;
             // If we haven't locked in first_fixation_ms, we just let it equal first_pass
             // until they leave the word. (A simplified heuristic).
             metric.first_fixation_ms += diff;
          }
          
          metric.lastGazeTime = time;
        }
        
        lastFixatedWordIdRef.current = fixatedWordId;
      }
    };

    window.addEventListener('gazeData', handleGaze);
    return () => {
      window.removeEventListener('gazeData', handleGaze);
    };
  }, [text]);

  const handleFinishReading = async () => {
    setIsAnalyzing(true);
    
    // Prepare data to send
    const wordsPayload = Object.values(metricsRef.current).map(m => ({
      word: m.word,
      word_id: m.word_id,
      first_fixation_ms: m.first_fixation_ms,
      first_pass_ms: m.first_pass_ms,
      fixation_count: m.fixation_count,
      regression_count: m.regression_count,
      skip_flag: m.skip_flag
    }));

    try {
      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ words: wordsPayload })
      });
      
      const data = await response.json();
      setResults(data.results);
    } catch (error) {
      console.error("Failed to predict:", error);
      alert("Backend prediction failed. Is FastAPI running on port 8000?");
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <div className="glass-panel">
      <h2>Read the text below normally</h2>
      
      <div className="reading-area" style={{ marginTop: '2rem', marginBottom: '3rem', textAlign: 'left' }}>
        {paragraphs.map((p, pIdx) => (
          <p key={pIdx} style={{ marginBottom: '1.5rem', lineHeight: '1.8' }}>
            {p.map((w) => {
              const res = results.find(r => r.word_id === w.id);
              const color = res ? getDifficultyColor(res.predicted_difficulty_ms) : 'inherit';
              
              return (
                <span 
                  key={w.id} 
                  className="word"
                  ref={el => wordElementsRef.current[w.id] = el}
                  style={{ color }}
                  data-tooltip={res ? `${Math.round(res.predicted_difficulty_ms)}ms` : ''}
                >
                  {w.word}&nbsp;
                </span>
              );
            })}
          </p>
        ))}
      </div>

      <button 
        className="btn" 
        onClick={handleFinishReading}
        disabled={isAnalyzing}
      >
        {isAnalyzing ? 'Analyzing...' : 'Finish Reading'}
      </button>

      {results.length > 0 && (
        <div style={{ marginTop: '2rem', display: 'flex', gap: '1rem', justifyContent: 'center' }}>
          <div style={{color: '#4ade80'}}>Easy</div>
          <div style={{color: '#facc15'}}>Medium</div>
          <div style={{color: '#fb923c'}}>Hard</div>
          <div style={{color: '#f87171'}}>Very Hard</div>
        </div>
      )}
    </div>
  );
}
