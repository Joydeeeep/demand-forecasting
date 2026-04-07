export default function MetricCards({ metrics, modelName }) {
  if (!metrics) {
    return (
      <div className="text-slate-400 text-center py-6">
        Loading metrics...
      </div>
    );
  }

  const formatNumber = (num) => {
    return Number(num).toLocaleString(undefined, {
      maximumFractionDigits: 2,
    });
  };

  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
      {Object.entries(metrics).map(([key, value]) => (
        <div
          key={key}
          className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-lg hover:scale-[1.02] transition"
        >
          <p className="text-slate-400 text-sm uppercase tracking-wide">
            {key}
          </p>
          <h3 className="text-2xl font-bold mt-2">
            {formatNumber(value)}
          </h3>
          <p className="text-slate-500 text-sm mt-1 capitalize">
            {modelName}
          </p>
        </div>
      ))}
    </div>
  );
}