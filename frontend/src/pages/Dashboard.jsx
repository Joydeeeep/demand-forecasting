import { useEffect, useState } from "react";

import Header from "../components/Header";
import ModelTable from "../components/ModelTable";
import MetricCards from "../components/MetricCards";
import PredictionChart from "../components/PredictionChart";
import ForecastPanel from "../components/ForecastPanel";

import {
  getModels,
  getComparison,
  getMetrics,
  getPredictions,
  getForecast,
  getActuals,
} from "../api/forecastApi";

export default function Dashboard() {
  const [models, setModels] = useState([]);
  const [comparison, setComparison] = useState([]);
  const [metrics, setMetrics] = useState(null);
  const [predictions, setPredictions] = useState(null);
  const [selectedModel, setSelectedModel] = useState("xgboost");
  const [loading, setLoading] = useState(true);
  const [actuals, setActuals] = useState(null);

  useEffect(() => {
    fetchInitialData();
  }, []);

 const fetchInitialData = async () => {
  try {
    setLoading(true);

    const modelData = await getModels();
    const comparisonData = await getComparison();

    const actualData = await getActuals();
    
    setModels(modelData.available_models);
    setComparison(comparisonData);
    setActuals(actualData.actuals);


    await loadModelData("xgboost");
  } catch (error) {
    console.error(error);
  } finally {
    setLoading(false);
  }
};

  const loadModelData = async (modelName) => {
    try {
      const metricsData = await getMetrics(modelName);
      const predictionData = await getPredictions(modelName);

      setSelectedModel(modelName);
      setMetrics(metricsData.metrics);
      setPredictions(predictionData.predictions);
    } catch (error) {
      console.error(`Error loading data for ${modelName}:`, error);
    }
  };

  const handleForecast = async (modelName) => {
    try {
      const forecastData = await getForecast(modelName);
      setSelectedModel(modelName);
      setPredictions(forecastData.predictions);

      const metricsData = await getMetrics(modelName);
      setMetrics(metricsData.metrics);
    } catch (error) {
      console.error(`Error forecasting with ${modelName}:`, error);
    }
  };

  if (loading) {
  return (
    <div className="min-h-screen flex items-center justify-center text-slate-400">
      Loading dashboard...
    </div>
  );
}

  return (
    <div className="min-h-screen bg-slate-950 text-white px-6 py-8 md:px-12">
      <div className="max-w-7xl mx-auto space-y-8">
        <Header />

        <ForecastPanel models={models} onForecast={handleForecast} />

        <MetricCards metrics={metrics} modelName={selectedModel} />

        <PredictionChart predictions={predictions} actuals={actuals} />

        <ModelTable data={comparison} onSelectModel={loadModelData} />
      </div>
    </div>
  );
}