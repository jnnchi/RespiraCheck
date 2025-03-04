import React from 'react';

const LoadingDots = () => {
    return (
        <div className="flex items-center gap-10 max-w-[597px]">
            <div className="h-8 w-8 bg-[#EED65C] rounded-full animate-bounce [animation-delay:-0.4s]"></div>
            <div className="h-8 w-8 bg-[#EED65C] rounded-full animate-bounce [animation-delay:-0.2s]"></div>
            <div className="h-8 w-8 bg-[#EED65C] rounded-full animate-bounce"></div>
        </div>
    );
}; export default LoadingDots;