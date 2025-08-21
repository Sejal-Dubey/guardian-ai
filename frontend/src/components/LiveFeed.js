import React from 'react';

const LiveFeed = ({ events, threatsToday, status, onAnalyze }) => {
    const statusClass = status === "Threat Neutralized" 
        ? "bg-gradient-to-r from-red-600 to-red-800 text-white" 
        : "bg-gray-800";

    const handleSubmit = (e) => {
        e.preventDefault();
        const formData = {
            sender: e.target.sender.value,
            subject: e.target.subject.value,
            body: e.target.body.value,
        };
        onAnalyze(formData);
    };

    return (
        <div className="bg-gray-900/50 p-6 rounded-2xl shadow-lg border border-gray-800 space-y-6 h-full flex flex-col">
            {/* --- NEW: Input Form --- */}
            <div>
                <h3 className="text-lg font-semibold mb-2 text-violet-300">Simulate Attack</h3>
                <form onSubmit={handleSubmit} className="space-y-3">
                    <input name="sender" type="email" required className="w-full bg-gray-800 border border-gray-700 rounded-md p-2 text-sm" placeholder="Sender Email" defaultValue="security@yourbank-alerts.com" />
                    <input name="subject" type="text" required className="w-full bg-gray-800 border border-gray-700 rounded-md p-2 text-sm" placeholder="Subject" defaultValue="Urgent Security Alert" />
                    <textarea name="body" required rows="3" className="w-full bg-gray-800 border border-gray-700 rounded-md p-2 text-sm" placeholder="Email Body">Dear Customer, urgent action is required on your account.</textarea>
                    <button type="submit" className="w-full bg-violet-600 hover:bg-violet-700 text-white font-bold py-2 px-4 rounded-md transition-colors duration-300">
                        Analyze
                    </button>
                </form>
            </div>
            
            <div className={`text-center p-4 rounded-lg ${statusClass} transition-colors duration-500`}>
                <p className="text-lg font-medium">{status}</p>
            </div>

            <div>
                <h3 className="text-lg font-semibold mb-2 text-violet-300">Live Event Feed</h3>
                <div className="space-y-3 h-48 overflow-y-auto pr-2">
                    {events.length === 0 ? (
                        <p className="text-sm text-gray-500">No events yet...</p>
                    ) : (
                        events.map((event, index) => (
                            <p key={index} className="text-sm"><strong>[{event.time}] {event.type} Event:</strong> Risk score {event.score}%</p>
                        ))
                    )}
                </div>
            </div>

            <div>
                <h3 className="text-lg font-semibold mb-2 text-violet-300">System Performance</h3>
                <div className="grid grid-cols-3 gap-4 text-center">
                    <div className="bg-gray-800 p-3 rounded-lg">
                        <p className="text-2xl font-bold text-violet-400">99.7%</p>
                        <p className="text-xs text-gray-500">Model Acc.</p>
                    </div>
                    <div className="bg-gray-800 p-3 rounded-lg">
                        <p className="text-2xl font-bold text-violet-400">&lt;150ms</p>
                        <p className="text-xs text-gray-500">Avg. Latency</p>
                    </div>
                    <div className="bg-gray-800 p-3 rounded-lg">
                        <p className="text-2xl font-bold text-violet-400">{threatsToday}</p>
                        <p className="text-xs text-gray-500">Threats Today</p>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default LiveFeed;