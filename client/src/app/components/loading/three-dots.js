"use client"
import React, { useState, useEffect } from 'react';
import ResultsButton from './button';

const LoadingDots = () => {
    const [isProcessing, setIsProcessing] = useState(true);

    useEffect(() => {
        // process completes after 2 seconds
        setTimeout(() => {
          setIsProcessing(false); 
        }, 2000);
      }, []);

    return (
      <div>
        {
          isProcessing ? (
            <div className="flex items-center gap-10 max-w-[597px]">
            <div className="h-8 w-8 bg-[#EED65C] rounded-full animate-bounce [animation-delay:-0.4s]"></div>
            <div className="h-8 w-8 bg-[#EED65C] rounded-full animate-bounce [animation-delay:-0.2s]"></div>
            <div className="h-8 w-8 bg-[#EED65C] rounded-full animate-bounce"></div>
        </div>
          ) : (
            <ResultsButton className="transition delay-100 duration-200 ease-in-out"></ResultsButton>
          )
        }
      </div>
    );
}; export default LoadingDots;