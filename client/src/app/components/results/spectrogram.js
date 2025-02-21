import { Box } from "@mui/material";
import React from "react";

const Spectrogram = () => {
    return (
        <Box sx = {{width: 479, height: 389, position: "relative"}}>
            <Box
                component="img" 
                sx={{ width: 479, height: 389, position: "absolute"}} 
                    alt="Image" 
                    src="https://via.placeholder.com/479x389"
                />
        </Box>
    )
}

export default Spectrogram;