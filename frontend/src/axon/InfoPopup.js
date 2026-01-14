// InfoPopup Component
import React, { useState, useEffect, forwardRef, useImperativeHandle } from "react";
import { Snackbar } from "@mui/material";

const InfoPopup = forwardRef((props, ref) => {
  const [popupOpen, setPopupOpen] = useState(false);
  const [popupMessage, setPopupMessage] = useState("");

  // Expose showMessage method to parent via ref
  useImperativeHandle(ref, () => ({
    showMessage: (message) => {
      setPopupMessage(message);  // Set the message
      setPopupOpen(true);  // Show the popup
    },
  }));

  // Auto-close the popup after 3 seconds
  useEffect(() => {
    if (popupOpen) {
      const timer = setTimeout(() => {
        setPopupOpen(false);
      }, 3000);
      return () => clearTimeout(timer);
    }
  }, [popupOpen]);

  return (
    <Snackbar
      open={popupOpen}
      autoHideDuration={null}
      message={popupMessage}
      anchorOrigin={{
        vertical: "bottom",
        horizontal: "right",
      }}
    />
  );
});

export default InfoPopup;


/*
Usage:
import React, { useRef } from "react";

  const infoPopupRef = useRef(null);
        infoPopupRef.current.showMessage("Some message"); 

*/