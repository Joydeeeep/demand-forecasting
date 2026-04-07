import { BarChart3 } from "lucide-react";

export default function Header() {
  return (
    <div className="flex items-center gap-4 mb-8">
      <div className="p-3 rounded-2xl bg-blue-500/20 border border-blue-400/30">
        <BarChart3 className="w-8 h-8 text-blue-400" />
      </div>
      <div>
        <h1 className="text-3xl font-bold tracking-tight">
          Demand Forecasting Dashboard
        </h1>
        <p className="text-slate-400 mt-1">
          Compare forecasting models, inspect metrics, and visualize predictions.
        </p>
      </div>
    </div>
  );
}