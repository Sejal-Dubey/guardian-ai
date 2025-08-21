import React, { useState, useEffect } from 'react';

const Header = ({ isConnected }) => {
    const [time, setTime] = useState(new Date().toLocaleTimeString('en-US'));

    useEffect(() => {
        const timer = setInterval(() => {
            setTime(new Date().toLocaleTimeString('en-US'));
        }, 1000);
        return () => clearInterval(timer);
    }, []);

    return (
        <div className="lg:col-span-3">
             <div className="text-center mb-8">
                <h1 className="text-4xl md:text-5xl font-bold text-violet-400">GuardianAI</h1>
                <div className="flex justify-center items-center space-x-2 mt-2">
                    <div className={`w-3 h-3 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'}`} title={isConnected ? "Connected" : "Disconnected"}></div>
                    <p className="text-sm text-gray-500">{time} - Pimpri-Chinchwad, India</p>
                </div>
            </div>
        </div>
    );
};

export default Header;