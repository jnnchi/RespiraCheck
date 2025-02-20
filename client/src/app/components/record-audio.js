import React from "react";
import Card from "@mui/material/Card";
import CardContent from "@mui/material/CardContent";
import MicIcon from '@mui/icons-material/Mic';

const RecordAudio = () => {
    return (
        <div className="flex flex-col items-center w-[370px] ">
        <h1 className="mb-5 [font-family:'Spartan-Bold',Helvetica] font-bold text-4xl text-center tracking-[0.15px] leading-[54px]">
          Record Audio:
        </h1>

        <Card className="w-full rounded-[40px] cursor-pointer" sx={{width: 350, height: 310, bgcolor: '#F1F7FF', transition: "transform 0.3s, box-shadow 0.3s", "&:hover": {transform: "scale(1.05)", boxShadow: '0px 2px 10px rgba(0, 0, 0, 0.1)',} }}>
          <CardContent className="flex flex-col items-center pt-1 px-10 bg-[#F1F7FF] space-y-9">
            <button
                className="pt-5 w-[200px] h-[200px] bg-[#ffd05c] rounded-full flex items-center justify-center"
                aria-label="Start recording"
            >
                <MicIcon className="pm-1 text-black" sx={{ width: 120, height: 120, color: '#ffffff' }} />
            </button>

            <p className="[font-family:'Spartan-Regular',Helvetica] font-normal text-xl text-center tracking-[0.15px] leading-[30px]" >
              Record your cough audio
            </p>
          </CardContent>

        </Card>
      </div>
    );
}; export default RecordAudio;