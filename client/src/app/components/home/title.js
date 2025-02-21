import { Box, Typography } from "@mui/material";
import React from "react";

const Title = () => {
    return (
        <Box sx={{maxWidth: 710, height: 161}}>

            <Typography
                sx={{
                    fontSize: 55,
                    fontWeight: 600, 
                    color: "black",
                    lineHeight: "82.5px",
                    letterSpacing: "0.15px",
                }}
            >
                Detect COVID-19 with the click of a button.
            </Typography>
        </Box>
        
    )
}

export default Title;
