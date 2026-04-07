import axios from "axios";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

export const getModels = async () => {
  const response = await axios.get(`${API_BASE_URL}/models`);
  return response.data;
};

export const getComparison = async () => {
  const response = await axios.get(`${API_BASE_URL}/comparison`);
  return response.data;
};

export const getMetrics = async (modelName) => {
  const response = await axios.get(`${API_BASE_URL}/metrics/${modelName}`);
  return response.data;
};

export const getPredictions = async (modelName) => {
  const response = await axios.get(`${API_BASE_URL}/predictions/${modelName}`);
  return response.data;
};

export const getForecast = async (modelName) => {
  const response = await axios.post(`${API_BASE_URL}/forecast`, {
    model_name: modelName,
  });
  return response.data;
};

export const getActuals = async () => {
  const response = await axios.get(`${API_BASE_URL}/actuals`);
  return response.data;
};