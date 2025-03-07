import { Box, Typography } from "@mui/material";
import React from "react";

const Result = () => {
    const prediction = localStorage.getItem("prediction");
    const results = {
        0: "No COVID-19",
        1: "COVID-19"
    }
    return (
        <>
        <Box sx={{ width: 616, height: 120 }}>
              <Typography
                variant="h1"
                sx={{
                  fontFamily: "'Spartan', sans-serif",
                  fontWeight: 300,
                  WebkitTextStroke: "0.5px black", 
                  color: "black",
                  fontSize: 40,
                  letterSpacing: 0.15,
                  lineHeight: "60px",
                }}
              >
                RespiraCheck has detected:
              </Typography>
        </Box>

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
                    lineHeight: "30px", 
                    whiteSpace: "nowrap",
                }}
            >
                {results[prediction]}
            </Typography>
        </Box>
        </>
    );
 
};

export default Result;