import React, { useState } from 'react';

const AnalysisDashboard = ({ data }) => {
    const [activeTab, setActiveTab] = useState('fusion');

    const renderTabContent = () => {
        if (!data) return <p className="text-gray-500 text-center mt-8">Submit an email and audio file to begin analysis.</p>;

        switch(activeTab) {
            case 'voice':
                return (
                    <div className="space-y-4 animate-fade-in">
                        <div className="bg-gray-800 p-4 rounded-lg">
                            <h4 className="font-semibold text-violet-300">Call Risk Score</h4>
                            <p className="text-5xl font-bold mt-2">{(data.voiceResult.risk_score * 100).toFixed(0)}%</p>
                        </div>
                        <div className="bg-gray-800 p-4 rounded-lg text-sm space-y-2">
                            {Object.entries(data.voiceResult.evidence).map(([key, value]) => <div key={key}><strong>{key}:</strong> {value}</div>)}
                        </div>
                    </div>
                );
            case 'email':
                 return (
                    <div className="space-y-4 animate-fade-in">
                        <div className="bg-gray-800 p-4 rounded-lg">
                            <h4 className="font-semibold text-violet-300">Email Risk Score</h4>
                            <p className="text-5xl font-bold mt-2">{(data.emailResult.risk_score * 100).toFixed(0)}%</p>
                        </div>
                        <div className="bg-gray-800 p-4 rounded-lg text-sm space-y-2">
                            {Object.entries(data.emailResult.evidence).map(([key, value]) => <div key={key}><strong>{key}:</strong> {value}</div>)}
                        </div>
                    </div>
                );
            case 'fusion':
            default:
                return (
                    <div className="space-y-4 animate-fade-in">
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                            <div className="bg-gray-800 p-4 rounded-lg">
                                <h4 className="font-semibold text-violet-300">Overall Fused Risk</h4>
                                <p className="text-5xl font-bold mt-2">{(data.finalScore * 100).toFixed(0)}%</p>
                            </div>
                            <div className="bg-gray-800 p-4 rounded-lg">
                                <h4 className="font-semibold text-violet-300">Correlation Evidence</h4>
                                <ul className="mt-2 space-y-1 text-sm">
                                    {data.boostApplied ? (
                                        <>
                                            <li className="flex items-center text-green-400">✔ Voice call flagged for spoofing.</li>
                                            <li className="flex items-center text-green-400">✔ Email shows signs of fraud.</li>
                                            <li className="flex items-center text-green-400">✔ Both channels reference same topic.</li>
                                        </>
                                    ) : <li className="text-gray-500">No correlated signals detected.</li>}
                                </ul>
                            </div>
                        </div>
                    </div>
                );
        }
    };

    const TabButton = ({ id, children }) => (
        <button 
            onClick={() => setActiveTab(id)}
            className={`py-4 px-1 text-center border-b-2 font-medium text-lg transition-colors duration-200 ${activeTab === id ? 'border-violet-400 text-violet-300' : 'border-transparent text-gray-500 hover:text-gray-300'}`}
        >
            {children}
        </button>
    );

    return (
        <div className="lg:col-span-2 bg-gray-900/50 p-6 rounded-2xl shadow-lg border border-gray-800">
            <div className="mb-4 border-b border-gray-700">
                <nav className="flex -mb-px space-x-6">
                    <TabButton id="fusion">Fusion & Recommendation</TabButton>
                    <TabButton id="voice">Voice Call</TabButton>
                    <TabButton id="email">Email</TabButton>
                </nav>
            </div>
            {renderTabContent()}
        </div>
    );
};

export default AnalysisDashboard;