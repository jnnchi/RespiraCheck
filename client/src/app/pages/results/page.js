"use client";
import React from 'react';
import Spectrogram from '@/app/components/results/spectrogram';
import NextSteps from '@/app/components/results/next-steps';
import TextHeader from '@/app/components/results/text-header';
import Result from '@/app/components/results/result';
import { Box, Stack, ThemeProvider, Typography } from "@mui/material";
import Link from 'next/link';
import Navbar from '../../components/navbar';

const Results = () => {
    return (
        <div>
            <Navbar></Navbar>
            <Stack direction="column" spacing={0} sx={{paddingLeft: "120px", paddingTop: "80px"}}>
                <Stack direction="row" spacing={12}>
                    <Spectrogram/>
                    
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
                <Box sx={{ marginTop: "-50px !important"}}><NextSteps sx={{ marginTop: "0px !important"}}/></Box>
                
            </Stack>
            
        </div>
    );
};
export default Results;