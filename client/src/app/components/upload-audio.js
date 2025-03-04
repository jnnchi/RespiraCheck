"use client";

import React, { useState, useRef } from "react";
import { Box, Button, Typography } from "@mui/material";

const UploadAudio = () => {
  const [file, setFile] = useState(null);
  const [error, setError] = useState(null);
  const fileInputRef = useRef(null);

  const validTypes = ["audio/wav", "audio/mpeg", "audio/webm"];

  const uploadFile = async (audioBlob, mimeType) => {
    try {
      const formData = new FormData();
      var extension = mimeType.split("/")[1].split(";")[0]; // Extract file extension

      if (extension == "mpeg") {
        extension = "mp3";
      }
      formData.append("file", audioBlob, `upload.${extension}`);
      
      console.log(`Uploading: upload.${extension}`);

      const response = await fetch("http://localhost:8000/upload_audio", {
          method: "POST",
          body: formData,
      });

      const prediction = await response.json();
      console.log("Server response:", prediction); 

      setFile(null);
      setError(null);
      console.log("Upload successful!");
    } catch (e) {
      console.error("Upload failed:", e);
      setError("Upload failed. Please try again.");
    }
  };

  const handleFileChange = (e) => {
    const selectedFile = e.target.files?.[0];

    if (selectedFile) {
      if (!validTypes.includes(selectedFile.type)) {
        setError("Invalid file type. Only .wav, .mp3, and .webm files are allowed.");
        return;
      }

      setFile(selectedFile);
      setError(null);

      const reader = new FileReader();
      reader.readAsArrayBuffer(selectedFile);
      reader.onloadend = () => {
        const audioBlob = new Blob([reader.result], { type: selectedFile.type });
        uploadFile(audioBlob, selectedFile.type);
      };
    }
  };

  return (
    <div className="flex flex-col items-center w-[370px]">
      <Typography variant="h4" fontWeight="bold" textAlign="center" mb={2}>
        Upload Audio:
      </Typography>

      {error && (
        <Typography color="error" textAlign="center" mb={2}>
          {error}
        </Typography>
      )}

      <Button
        type="button"
        className="w-full rounded-[40px] cursor-pointer"
        sx={{
          borderRadius: "25px",
          textTransform: "none",
          color: "black",
          width: 350,
          height: 310,
          bgcolor: "#F1F7FF",
          transition: "transform 0.3s, box-shadow 0.3s",
          "&:hover": {
            transform: "scale(1.05)",
            boxShadow: "0px 2px 10px rgba(0, 0, 0, 0.1)",
          },
        }}
        onClick={() => fileInputRef.current.click()}
      >
        <Box className="flex flex-col items-center pt-1 px-10 bg-[#F1F7FF]">
          <img
            className="w-[230px] h-50 mb-8"
            alt="Upload files illustration"
            src="/upload-file.png"
          />
          <Typography variant="body1" textAlign="center">
            Upload your cough audio (.wav, .mp3, .webm)
          </Typography>
          {file && (
            <Typography variant="caption" color="textSecondary">
              Selected: {file.name}
            </Typography>
          )}
        </Box>
      </Button>

      {/* Hidden file input */}
      <input
        ref={fileInputRef}
        type="file"
        name="file"
        accept=".wav,.mp3,.webm"
        style={{ display: "none" }}
        onChange={handleFileChange}
      />
    </div>
  );
};

export default UploadAudio;