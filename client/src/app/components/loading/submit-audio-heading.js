import React from "react";

const SubmitAudioHeading = () => {
  return (
    <div className="flex items-center gap-4 max-w-[697px] transition ease-in-out delay-100 duration-200 hover:scale-110">
      <div className="relative w-12 h-[46px]">
        {/* Base circle */}
        <div className="absolute inset-0 bg-[#83a2ee] rounded-[24px/23px]" />
        {/* Overlay circle */}
        <div className="absolute inset-0 bg-[#3d70ec] rounded-[24px/23px]" />
        {/* Layered numbers for effect */}
        <div className="absolute inset-0 flex items-center justify-center">
          <span className="text-white [font-family:'Roboto-SemiBold',Helvetica] font-semibold text-[32px] leading-[48px] tracking-[0.15px]">
            1
          </span>
        </div>
        <div className="absolute inset-0 flex items-center justify-center">
          <span className="text-[#f1e39b] [font-family:'Roboto-SemiBold',Helvetica] font-semibold text-[32px] leading-[48px] tracking-[0.15px]">
            1
          </span>
        </div>
      </div>

      <h1 className="[font-family:'Spartan-Bold',Helvetica] font-bold text-black text-[50px] tracking-[0.15px] leading-[75px]">
        Submit Cough Audio
      </h1>
    </div>
  );
};
export default SubmitAudioHeading;
