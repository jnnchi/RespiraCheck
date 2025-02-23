import React from "react";
import { Box, Button} from "@mui/material";

const UploadAudio = () => {
    return (
        <div className="flex flex-col items-center w-[370px]">
        <h1 className="mb-5 [font-family:'Spartan-Bold',Helvetica] font-bold text-4xl text-center tracking-[0.15px] leading-[54px]">
          Upload Audio:
        </h1>

        <Button className="w-full rounded-[40px] cursor-pointer" sx={{textTransform: "none", color: "black", width: 350, height: 310, bgcolor: '#F1F7FF', transition: "transform 0.3s, box-shadow 0.3s", "&:hover": {transform: "scale(1.05)", boxShadow: '0px 2px 10px rgba(0, 0, 0, 0.1)',} }}>
          <Box className="flex flex-col items-center pt-1 px-10 bg-[#F1F7FF]">
            <img
              className="w-[230px] h-50 mb-8"
              alt="Upload files illustration"
              src="/upload-file.png"
            />

            <p className="[font-family:'Spartan-Regular',Helvetica] font-normal text-xl text-center tracking-[0.15px] leading-[30px]">
              Upload your cough audio
            </p>
          </Box>
        </Button>
      </div>
    );
}; export default UploadAudio;