import axios from "axios";

const hostIP = window.location.hostname;
const hostPort = window.location.port || 4000;
const protocol = window.location.protocol;

export const api = axios.create({
  baseURL: `${protocol}//${hostIP}:${hostPort}/imswitch/api/FileManager`,
});
