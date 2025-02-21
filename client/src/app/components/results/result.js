import { Box, Typography } from "@mui/material";
import React from "react";

const Result = () => {
    return (
        <Box position="relative" width={352} height={75}>
            <Typography
                variant="h1"
                sx={{
                    position: "absolute", 
                    width: 214, 
                    top: 6, 
                    left: -1, 
                    WebkitTextStroke: "1px #3d70ec", 
                    fontFamily: "'Spartan', Helvetica", 
                    fontWeight: "bold", 
                    color: "#3d70ec",
                    fontSize: 40, 
                    letterSpacing: 0.15, 
                    lineHeight: "60px", 
                    whiteSpace: "nowrap",
                }}
            >
                COVID-19
            </Typography>
        </Box>
    );
};

export default Result;