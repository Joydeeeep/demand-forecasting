export default function ModelTable({ data, onSelectModel }) {
  if (!data || data.length === 0) return null;

  // find best model (lowest MAPE)
  const bestModel = data.reduce((best, curr) =>
    curr.MAPE < best.MAPE ? curr : best
  );

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-lg">
      <h2 className="text-xl font-semibold mb-4">🏆 Model Leaderboard</h2>

      <table className="w-full text-left">
        <thead>
          <tr className="text-slate-400 border-b border-slate-800">
            <th className="py-3">Model</th>
            <th className="py-3">MAPE</th>
            <th className="py-3">MAE</th>
            <th className="py-3">RMSE</th>
            <th className="py-3">Action</th>
          </tr>
        </thead>
        <tbody>
          {data.map((row, idx) => {
            const isBest = row.model === bestModel.model;

            return (
              <tr
                key={idx}
                className={`border-b border-slate-800/50 ${
                  isBest ? "bg-emerald-900/20" : ""
                }`}
              >
                <td className="py-3 font-medium">
                  {row.model}
                  {isBest && (
                    <span className="ml-2 text-xs bg-emerald-600 px-2 py-1 rounded">
                      BEST
                    </span>
                  )}
                </td>
                <td className="py-3">{row.MAPE}</td>
                <td className="py-3">{row.MAE}</td>
                <td className="py-3">{row.RMSE}</td>
                <td className="py-3">
                  <button
                    onClick={() => onSelectModel(row.model.toLowerCase())}
                    className="px-4 py-2 rounded-xl bg-blue-600 hover:bg-blue-500 transition"
                  >
                    View
                  </button>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}