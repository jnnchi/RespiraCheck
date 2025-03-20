import React from "react";
import CheckIcon from "@mui/icons-material/Check";

const FileStatus = () => {
  return (
    <div className="flex flex-col items-center w-[643px] h-[71px]">
      <p className="w-[643px] top-0 left-0 font-normal [font-family:'Spartan-Bold',Helvetica] text-black text-xl text-center tracking-[0.15px] leading-[30px] transition ease-in-out delay-100 duration-200 hover:scale-110">
        <span className="tracking-[0.03px]">Your audio file has been</span>
        <span className="[font-family:'Spartan-Bold',Helvetica] font-bold tracking-[0.03px]">
          {" "}
          received
        </span>
      </p>

      <div>
        <CheckIcon
          className="w-[82px] h-[83px] top-[4px] left-[289px] bg-[#e8f5ff] rounded-[60px] flex flex-col items-center justify-center shadow-md"
          sx={{ width: 115, height: 115, color: "#3D70EC", mt: 3 }}
        />
      </div>
    </div>
  );
};
export default FileStatus;
