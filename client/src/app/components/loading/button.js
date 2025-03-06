"use client"

import React from "react";
import { Box, Button, Typography } from "@mui/material";
import { useRouter } from 'next/navigation';


const ResultsButton = () => {

    const router = useRouter();

    const handleNavigate = () => {
        router.push('/pages/results'); 
    };

    return (
        <Button
        sx={{
            width: 227,
            height: 72, 
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
            letterSpacing: "0.15px",
            transition: "transform 0.3s",
            "&:hover": {
              transform: "scale(1.03)",
              boxShadow: '0px 6px 6px #00000040',
            }
        }}
        onClick={handleNavigate}
        >
            See Results
        </Button>
    )
}

export default ResultsButton;