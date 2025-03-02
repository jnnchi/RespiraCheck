import React from "react";
import { Typography, Box, Stack } from "@mui/material";
import { decodeAction } from "next/dist/server/app-render/entry-base";

const TextBoxes = () => {
    return (
        <Stack direction="column" spacing= {2} sx={{width: 776}}>
            <Typography sx= {{fontWeight: 300, color: "black", fontSize: "22pt" }}>
            RespiraCheck aims to provide a fast, accessible, and non-invasive screening 
            method for people who may not have immediate access to traditional COVID-19 tests. 
            By leveraging audio processing techniques such as vocal separation, noise reduction, 
            and bandpass filtering, RespiraCheck preserves diagnostic accuracy while providing 
            a convenient platform for you!
            </Typography>
            <Typography sx= {{fontWeight: 300, color: "black", fontSize: "22pt" }}>
                We use the crowdsourced COUGHVID dataset containing 2,800 labeled samples to
                train our model, applying data augmentation techniques such as time and frequency 
                masking to ensure we have a balanced number of positive and negative samples. 
                From there, we built a robust audio processing pipeline to reduce background 
                noise and filter out voices.
            </Typography>

        </Stack>
        
    );
};

export default TextBoxes;