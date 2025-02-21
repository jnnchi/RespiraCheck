import { Box, Typography } from "@mui/material";
import React from "react";

const NextSteps = () => {
    return (
        <Box sx={{ position: "relative", widht: "100%", maxWidth: 1123, height: 212}}>
            <Typography
                variant="h2"
                component="div"
                sx={{
                    position: "absolute",
                    top: 0,
                    left: 0,
                    fontFamily: "Spartan-SemiBold, Helvetica",
                    fontWeight: "bold",
                    color: "black",
                    lineHeight: "45px",
                }}
            >
                Next Steps:
            </Typography>

            <Typography
                variant="body1"
                component="p"
                sx={{
                    position: "absolute",
                    top: 61, 
                    left: 0, 
                    fontFamily: "Spartan-Regular, Helvetica",
                    color: "black",
                    lineHeight: "30px",
                }}
            >
                Some text here, varies on result. 
            </Typography>
        </Box>
    );
};

export default NextSteps;