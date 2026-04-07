import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  ResponsiveContainer,
  Legend,
} from "recharts";

export default function PredictionChart({ predictions, actuals }) {
  if (!predictions || !actuals) {
    return (
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-lg text-slate-400">
        Loading chart...
      </div>
    );
  }

  const chartData = Object.keys(predictions).map((date) => {
    const cleanDate = String(date).split(" ")[0];

    return {
      date: cleanDate,
      predicted: Number(predictions[date]),
      actual:
        actuals[cleanDate] !== undefined
          ? Number(actuals[cleanDate])
          : null,
    };
  });

  console.log("Predictions:", predictions);
  console.log("Actuals:", actuals);
  console.log("Chart Data:", chartData);

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-lg">
      <h2 className="text-xl font-semibold mb-4">📈 Predicted vs Actual</h2>

      <div className="h-[350px]">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
            <XAxis dataKey="date" stroke="#94a3b8" />
            <YAxis stroke="#94a3b8" />
            <Tooltip />
            <Legend />
            <Line
              type="monotone"
              dataKey="actual"
              stroke="#22c55e"
              strokeWidth={3}
              dot={true}
              name="Actual"
            />
            <Line
              type="monotone"
              dataKey="predicted"
              stroke="#3b82f6"
              strokeWidth={3}
              dot={true}
              name="Predicted"
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}