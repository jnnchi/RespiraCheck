import { Box, Typography } from "@mui/material";
import React from "react";

const NextSteps = () => {
    return (
        <Box sx={{ paddingTop: 0, position: "relative", width: "100%", maxWidth: 1123, height: 0}}>
            <Typography
                variant="h2"
                component="div"
                sx={{
                    position: "relative",
                    top: 0,
                    left: 0,
                    fontFamily: "Spartan-SemiBold, Helvetica",
                    fontWeight: 500,
                    color: "black",
                    fontSize: "30px",
                }}
            >
                Next Steps:
            </Typography>

            <Typography
                variant="body1"
                component="p"
                sx={{
                    position: "relative",
                    top: 0, 
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