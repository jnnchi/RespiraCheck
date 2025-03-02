"use client"

import React from "react";
import { Box, Button, Typography } from "@mui/material";
import { useRouter } from 'next/navigation';


const StartButton = () => {

    const router = useRouter();

    const handleNavigate = () => {
        router.push('pages/use-the-tool'); 
    };

    return (
        <Button
        sx={{
            width: 257,
            height: 92, 
            backgroundColor: "#3d70ec",
            borderRadius: 30, 
            boxShadow: "0px 4px 4px #00000040",
            position: "relative",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            fontWeight: 600, 
            fontSize: 30, 
            color: "white",
            letterSpacing: "0.1em",
            textTransform: "none",
            letterSpacing: "0.15px"
        }}
        onClick={handleNavigate}
        >
            Get Started
        </Button>
    )
}

export default StartButton;