import React from 'react';

import Navbar from '../../components/navbar';
import TextBoxes from '@/app/components/about/text-boxes';
import Image from 'next/image';
import { Box} from "@mui/material";



const Info = () => {
    return (
        <div>
            <Navbar></Navbar>
            <Box sx={{paddingTop: "100px", paddingLeft: "120px"}}>
                <TextBoxes/>
            </Box>

            <Box sx={{ 
                paddingTop: "100px", 
                paddingLeft: "120px", 
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