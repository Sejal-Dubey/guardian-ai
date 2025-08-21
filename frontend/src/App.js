import React, { useState, useEffect } from 'react';
import io from 'socket.io-client';
import LiveFeed from './components/LiveFeed';
import AnalysisDashboard from './components/AnalysisDashboard';
import AlarmOverlay from './components/AlarmOverlay';
import Header from './components/Header';

const SOCKET_SERVER_URL = "http://127.0.0.1:8000";

function App() {
  const [analysisData, setAnalysisData] = useState(null);
  const [isConnected, setIsConnected] = useState(false);
  const [threatsToday, setThreatsToday] = useState(0);
  const [liveFeedEvents, setLiveFeedEvents] = useState([]);
  const [showAlarm, setShowAlarm] = useState(false);

  useEffect(() => {
    const socket = io(SOCKET_SERVER_URL, { transports: ["websocket"] });

    socket.on("connect", () => setIsConnected(true));
    socket.on("disconnect", () => setIsConnected(false));

    socket.on("analysis_result", (data) => {
      setAnalysisData(data);
      const now = new Date().toLocaleTimeString('en-US');
      const newEvents = [
          { time: now, type: 'Email', score: (data.emailResult.risk_score * 100).toFixed(0) },
          { time: now, type: 'Voice', score: (data.voiceResult.risk_score * 100).toFixed(0) }
      ];
      setLiveFeedEvents(prevEvents => [...newEvents, ...prevEvents]);

      if (data.finalScore > 0.7) {
        setShowAlarm(true);
        setThreatsToday(prev => prev + 1);
      }
    });

    return () => socket.disconnect();
  }, []);

  // --- NEW: Function to handle form submission ---
  const handleAnalysisRequest = async (formData) => {
    try {
        await fetch(`${SOCKET_SERVER_URL}/analyze`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(formData),
        });
    } catch (error) {
        console.error("Failed to send analysis request:", error);
    }
  };

  return (
    <div className="bg-gray-900 min-h-screen text-gray-300 font-sans p-4 md:p-6">
      <div className="container mx-auto max-w-7xl">
        <Header isConnected={isConnected} />
        
        <div className="flex flex-col lg:flex-row gap-6 mt-8">
          <div className="lg:w-1/3">
            <LiveFeed 
              events={liveFeedEvents} 
              threatsToday={threatsToday} 
              status={analysisData ? (analysisData.finalScore > 0.7 ? "Threat Neutralized" : "Low Risk Detected") : "Awaiting Analysis"}
              onAnalyze={handleAnalysisRequest} // Pass the handler function
            />
          </div>
          <div className="lg:w-2/3">
            <AnalysisDashboard data={analysisData} />
          </div>
        </div>
      </div>
      <AlarmOverlay show={showAlarm} onClose={() => setShowAlarm(false)} />
    </div>
  );
}

export default App;