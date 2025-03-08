import React from 'react';

import Navbar from '../../components/navbar';
import TextBoxes from '@/app/components/about/text-boxes';
import Image from 'next/image';
import { Box, Stack } from "@mui/material";
import AboutHeader from '@/app/components/about/header';

const Info = () => {
    return (
        <div>
            <Navbar></Navbar>

            <Stack direction="column" spacing={4} sx={{paddingTop: "4vw", paddingLeft: "20vw"}}>
                <AboutHeader></AboutHeader>
                <TextBoxes/>
            </Stack>

            <Box sx={{ 
                paddingTop: "16vw", 
                paddingLeft: "22vw", 
                position: "absolute", 
                width: 1300, 
                height: 400, 
                top: 180, 
                left: 0, 
                zIndex: 1}}>
                <Image src="/undraw_doctor.svg" alt="doctor image" width={1267} height={351}/> 
            </Box>

        </div>
    );
};
export default Info;