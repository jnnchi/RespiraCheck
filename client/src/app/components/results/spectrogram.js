import { Box } from "@mui/material";
import React from "react";

const Spectrogram = () => {
    return (
        <Box sx = {{width: 479, height: 389, position: "relative"}}>
            <Box
                component="img" 
                sx={{width: 479, height: 300, position: "absolute", backgroundColor: "black"}} 
                />
        </Box>
    )
}

export default Spectrogram;