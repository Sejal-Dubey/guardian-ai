import React, { useEffect } from 'react';

// This function creates the audible alarm sound
function playAlarm() {
    const audioContext = new (window.AudioContext || window.webkitAudioContext)();
    if (audioContext.state === 'suspended') {
        audioContext.resume();
    }
    const oscillator = audioContext.createOscillator();
    const gainNode = audioContext.createGain();
    oscillator.connect(gainNode);
    gainNode.connect(audioContext.destination);
    oscillator.type = 'sine';
    oscillator.frequency.setValueAtTime(1200, audioContext.currentTime);
    gainNode.gain.setValueAtTime(0.8, audioContext.currentTime);
    oscillator.start();
    oscillator.stop(audioContext.currentTime + 1.5);
}

const AlarmOverlay = ({ show, onClose }) => {
    useEffect(() => {
        if (show) {
            playAlarm();
        }
    }, [show]);

    if (!show) return null;

    return (
        <div className="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50 animate-fade-in">
            <div className="bg-gray-800 border-4 border-red-600 rounded-2xl p-8 text-center max-w-md w-full shadow-2xl animate-pulse">
                <svg className="mx-auto h-20 w-20 text-red-500" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth="1.5" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" d="M15.75 10.5l4.72-4.72a.75.75 0 011.28.53v11.38a.75.75 0 01-1.28.53l-4.72-4.72M4.5 12a7.5 7.5 0 0015 0m-15 0a7.5 7.5 0 1115 0m-15 0H3m16.5 0H21m-1.5 0H12m-6 0h3.75" />
                    <path strokeLinecap="round" strokeLinejoin="round" d="M3 3l18 18" />
                </svg>
                <h2 className="text-4xl font-bold text-red-500 mt-4">THREAT DETECTED</h2>
                <p className="text-2xl text-white mt-2">HANG UP THE CALL IMMEDIATELY</p>
                <p className="text-gray-400 mt-4">A coordinated fraud attempt has been identified. Do not share any personal information.</p>
                <button onClick={onClose} className="mt-8 bg-red-600 hover:bg-red-700 text-white font-bold py-3 px-8 rounded-lg text-lg">
                    Dismiss
                </button>
            </div>
        </div>
    );
};

export default AlarmOverlay;
