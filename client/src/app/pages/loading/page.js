import React from 'react';

import Card from "@mui/material/Card";
import CardContent from "@mui/material/CardContent";
import { Box, Typography, Stack } from "@mui/material";
import { ThemeProvider } from "@mui/material/styles";
import theme from "../../theme/theme";

import SubmitAudioHeading from '@/app/components/submit-audio-heading';
import PredictingHeading from '@/app/components/predicting-heading';
import FileStatus from '@/app/components/file-status';
import Navbar from '@/app/components/navbar';
import LoadingDots from '@/app/components/three-dots';

export default function Loading() {
    return (
        <ThemeProvider theme={theme}>
            <Navbar></Navbar>
            <Stack width= "100%" direction ="column" alignItems="center" spacing={17} sx={{ justifyContent: "center", mt: 6 }} >

                <Stack width= "100%" direction ="column" alignItems="center" spacing={3} sx={{ justifyContent: "center", mb: 3 }} >
                    <SubmitAudioHeading></SubmitAudioHeading>
                    <FileStatus></FileStatus>
                </Stack>
                
                <Stack width= "100%" direction ="column" alignItems="center" spacing={9} sx={{ justifyContent: "center" }} >
                    <PredictingHeading></PredictingHeading>
                    <LoadingDots></LoadingDots>
                </Stack>

            </Stack>

            {/* <Card className="relative w-[705px] h-[533px] bg-transparent">
                <CardContent className="p-0">
                    <SubmitAudioHeading></SubmitAudioHeading>
                    <FileStatus></FileStatus>
                    <PredictingHeading></PredictingHeading>
                </CardContent>
            </Card> */}

        </ThemeProvider>
    );
};