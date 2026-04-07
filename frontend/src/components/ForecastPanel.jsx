import { useState } from "react";

export default function ForecastPanel({ models, onForecast }) {
  const [selectedModel, setSelectedModel] = useState("xgboost");

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-lg">
      <h2 className="text-xl font-semibold mb-4">🔮 Forecast Viewer</h2>

      <div className="flex flex-col md:flex-row gap-4">
        <select
          value={selectedModel}
          onChange={(e) => setSelectedModel(e.target.value)}
          className="bg-slate-800 border border-slate-700 rounded-xl px-4 py-3 text-white"
        >
          {models.map((model, idx) => (
            <option key={idx} value={model}>
              {model}
            </option>
          ))}
        </select>

        <button
          onClick={() => onForecast(selectedModel)}
          className="px-5 py-3 rounded-xl bg-emerald-600 hover:bg-emerald-500 transition"
        >
          Load Forecast
        </button>
      </div>
    </div>
  );
}