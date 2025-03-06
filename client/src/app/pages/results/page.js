"use client";
import React, { useEffect, useState } from "react";
import Spectrogram from '@/app/components/results/spectrogram';
import NextSteps from '@/app/components/results/next-steps';
import TextHeader from '@/app/components/results/text-header';
import Result from '@/app/components/results/result';
import { Box, Stack, ThemeProvider, Typography } from "@mui/material";
import Link from 'next/link';
import Navbar from '../../components/navbar';

const Results = () => {
    const [prediction, setPrediction] = useState(null);
    const [spectrogramImage, setSpectrogramImage] = useState(null);

    useEffect(() => {
        // This code runs only on the client
        const storedPrediction = localStorage.getItem("prediction");
        const storedImage = localStorage.getItem("spectrogram_image");

        console.log("Type of spectrogram_image:", typeof storedImage);
        if (storedPrediction && storedImage) {
          setPrediction(storedPrediction);
          setSpectrogramImage(storedImage);
        }
    }, []);

    return (
        <div>
            <Navbar></Navbar>
            <Stack direction="column" spacing={0} sx={{paddingLeft: "12vw", paddingTop: "80px"}}>
                <Stack direction="row" spacing={12}>
                    <img
                        src={`data:image/png;base64,${spectrogramImage}`}
                        alt="Spectrogram"
                        style={{width: '30vw', height: 'auto', position: "relative"}}
                    />
                    
                    <Stack direction="column" spacing={2}>
                        <TextHeader/>
                        <Result/>
                            <Typography 
                                sx={{
                                    position: "relative", 
                                    fontFamily: "'Spartan-Regular', Helvetica", 
                                    color: "#303030", 
                                    fontSize: "1.5rem",
                                    textDecoration: "underline",
                                    fontWeight: 200,
                                }
                            }>
                                <Link href="/pages/about"> 
                                    <span style={{ textDecoration: "none"}}>Learn more about our model&gt;</span>
                                </Link>

                            </Typography>
                    </Stack>
                </Stack>
                <Box sx={{ marginTop: "40px !important"}}>
                    <NextSteps sx={{ marginTop: "0px !important"}}/>
                </Box>
                
            </Stack>
            
        </div>
    );
};
export default Results;