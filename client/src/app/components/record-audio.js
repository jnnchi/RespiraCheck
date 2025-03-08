"use client";

import { React, useState, useRef } from "react";
import { Box, Card, CardContent, Button } from "@mui/material";
import MicIcon from "@mui/icons-material/Mic";
import { redirect } from "next/navigation";

// https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getUserMedia
// https://developer.mozilla.org/en-US/docs/Web/API/MediaRecorder

const RecordAudio = () => {
  const [recording, setRecording] = useState(false);
  // keeps track of MediaRecorder object + audio chunks
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);

  const startRecording = async () => {
    // accesses user mic (and waits for user permissions)
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const mediaRecorder = new MediaRecorder(stream);
    mediaRecorderRef.current = mediaRecorder;

    // pushes recorded data into audioChunksRef, given that audio has been recorded
    mediaRecorder.ondataavailable = (event) => {
      if (event.data.size > 0) {
        audioChunksRef.current.push(event.data);
      }
    };

    // saves audio, sends POST request to backend
    mediaRecorder.onstop = async () => {
      const audioBlob = new Blob(audioChunksRef.current, {
        type: mediaRecorderRef.current.mimeType,
      });
      await uploadAudio(audioBlob);
      audioChunksRef.current = [];
    };

    mediaRecorder.start();
    setRecording(true);
  };

  const stopRecording = () => {
    mediaRecorderRef.current?.stop();
    setRecording(false);
  };

  const uploadAudio = async (audioBlob) => {
    const formData = new FormData();
    const mimeType = mediaRecorderRef.current.mimeType; // e.g., "audio/webm"
    const extension = mimeType.split("/")[1].split(";")[0]; // "webm"
    formData.append("file", audioBlob, `recording.${extension}`);
    console.log(`recording.${extension}`);
    const response = await fetch("http://localhost:8000/upload_audio", {
      method: "POST",
      body: formData,
    });

    const result = await response.json();
    console.log("Server response:", result);

    localStorage.setItem("prediction", result.prediction);
    localStorage.setItem("spectrogram_image", result.spectrogram_image);

    // Redirect to the /loading page
    //router.push("/pages/results");
    redirect("/pages/loading");
  };

  const handleRecording = () => {
    if (recording) {
      stopRecording();
    } else {
      startRecording();
    }
  };

  return (
    <div className="flex flex-col items-center w-[370px] ">
      <h1 className="mb-5 [font-family:'Spartan-Bold',Helvetica] font-bold text-4xl text-center tracking-[0.15px] leading-[54px]">
        Record Audio:
      </h1>

      <Button
        onClick={handleRecording}
        className="w-full rounded-[40px] cursor-pointer"
        sx={{
          borderRadius: "25px",
          fontWeight: 400,
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
      >
        <Box className="flex flex-col items-center pt-1 px-10 bg-[#F1F7FF] space-y-9">
          <div
            className={`pt-5 w-[200px] h-[200px] bg-[#EED65C] rounded-full flex items-center justify-center ${
              recording ? "border-4 border-red-500 animate-pulse" : ""
            }`}
          >
            <MicIcon
              className="pm-1 text-black"
              sx={{ width: 120, height: 120, color: "#ffffff" }}
            />
          </div>

          <p className="[fontfont-normal text-xl text-center tracking-[0.15px] leading-[30px]">
            Record your cough audio
          </p>
        </Box>
      </Button>
    </div>
  );
};
export default RecordAudio;
