import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import App from './App';
import reportWebVitals from './reportWebVitals';

//redux state
import { Provider } from "react-redux"; // Import Provider from react-redux
import { PersistGate } from "redux-persist/integration/react";
import store, { persistor } from "./state/store"; // Import your store


const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  // REMOVED React.StrictMode - causes duplicate socket connections in development
  // StrictMode intentionally double-mounts components to catch side effects
  // This conflicts with WebSocket singleton pattern
  <Provider store={store}>
    <PersistGate loading={null} persistor={persistor}>
      <App />
    </PersistGate>
  </Provider>
);

// If you want to start measuring performance in your app, pass a function
// to log results (for example: reportWebVitals(console.log))
// or send to an analytics endpoint. Learn more: https://bit.ly/CRA-vitals
reportWebVitals();
