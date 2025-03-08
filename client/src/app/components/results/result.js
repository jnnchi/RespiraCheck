import { Box, Typography } from "@mui/material";
import React from "react";

const Result = ({prediction}) => {
    const diagnosis = 
        prediction === "1" ? "COVID-19" :"Do Not Have COVID-19";
    const color = prediction === "1" ? "#3d70ec" : "#3CB371";

    return (
        <Box position="relative" width={352} height={75}>
            <Typography
                variant="h1"
                sx={{
                    position: "absolute", 
                    width: 214, 
                    top: 6, 
                    left: -1, 
                    color: `${color}`, 
                    fontFamily: "'Spartan', Helvetica", 
                    fontWeight: "bold", 
                    fontSize: 40, 
                    letterSpacing: 0.15, 
                    lineHeight: "30px", 
                    whiteSpace: "nowrap",
                }}
            >
                {diagnosis}
            </Typography>
        </Box>
    );
};

export default Result;