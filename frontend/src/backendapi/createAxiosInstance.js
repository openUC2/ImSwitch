// src/hooks/useApiBaseUrl.js
import axios from 'axios'; 
import store from '../state/store';

const createAxiosInstance = () => {
  //get settings
  const state = store.getState(); 
  const rootPath = '/imswitch/api';
  //create instance
  return axios.create({
    baseURL: `${state.connectionSettingsState.ip}:${state.connectionSettingsState.apiPort}${rootPath}`,
  });
};

export default createAxiosInstance;
