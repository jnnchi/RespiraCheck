import { Box } from "@mui/material";
import React from "react";

const Spectrogram = (image) => {
  return (
    <Box sx={{ width: 479, height: 389, position: "relative" }}>
      <img
        style={{ width: "479px", height: "300px", position: "absolute" }}
        src={`data:image/png;base64,${image}`}
        alt="Spectrogram"
      />
      {/*<Box
                component="img" 
                sx={{width: 479, height: 300, position: "absolute", backgroundColor: "black"}} 
                />*/}
    </Box>
  );
};

export default Spectrogram;
