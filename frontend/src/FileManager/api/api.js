import axios from "axios";
import store from "../../state/store";

export function getFileManagerBaseUrl() {
  const s = store.getState().connectionSettingsState;
  return `${s.ip}:${s.apiPort}/imswitch/api/FileManager`;
}

export const api = axios.create();

api.interceptors.request.use((config) => {
  config.baseURL = getFileManagerBaseUrl();
  return config;
});
