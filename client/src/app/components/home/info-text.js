import React from "react";
import { Box, Typography } from "@mui/material";
import Link from 'next/link';


const InfoText = () => {
    return (
        <Box
            sx={{
                width: 600, 
                height: 218, 
                position: "relative",
            }}>
            <Typography
                sx={{
                    fontWeight: 300,
                    color: "black",
                    textJustify: "left",
                    fontSize: 19,
                }}>
                At RespiraCheck, our mission is to bring free, accurate COVID-19 testing to lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris. 
            </Typography>
            <Typography
                sx={{
                    paddingTop: "10px",
                    fontWeight: 100,
                    color: "black",
                    textJustify: "left",
                    fontSize: 19,
                    textDecoration: 'underline',
                    textDecorationThickness: 0.8,
                }}>

                <Link href="/pages/about"> 
            
                    <span style={{ textDecoration: "none"}}>Learn More &gt;</span>
                </Link>
                
            </Typography>
        </Box>
    )
}

export default InfoText;