import React from 'react';
import Spectrogram from '@/app/components/results/spectrogram';
import NextSteps from '@/app/components/results/next-steps';
import TextHeader from '@/app/components/results/text-header';
import Result from '@/app/components/results/result';
import { Box, Stack, Typography } from "@mui/material";

import Navbar from '../../components/navbar';

const Results = () => {
    return (
        <div>
            <Navbar></Navbar>
            <Stack direction="row" spacing={15} sx={{padding:"120px"}}>
                <Spectrogram/>
                <Stack direction="column" spacing={2}>
                    <TextHeader/>
                    <Result/>
                    <Typography 
                        sx={{
                            position: "relative", 
                            fontFamily: "'Spartan-Regular', Helvetica", 
                            color: "#303030", 
                            fontSize: "1.125rem",
                            textDecoration: "underline"
                        }
                    }>
                    Learn more about our model &gt;
                    </Typography>
                </Stack>

            </Stack>
            
        </div>
    );
};
export default Results;