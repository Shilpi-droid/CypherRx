// src/components/ConfidenceGauge.tsx
import React from 'react';
import { PieChart, Pie, Cell, ResponsiveContainer } from 'recharts';

interface ConfidenceGaugeProps {
  confidence: number;
}

export const ConfidenceGauge: React.FC<ConfidenceGaugeProps> = ({ confidence }) => {
  const percentage = confidence * 100;
  const data = [
    { value: percentage },
    { value: 100 - percentage }
  ];

  const getColor = () => {
    if (confidence > 0.7) return '#10b981'; // green
    if (confidence > 0.4) return '#f59e0b'; // yellow
    return '#ef4444'; // red
  };

  return (
    <div className="bg-white rounded-2xl shadow-xl p-6">
      <h3 className="text-lg font-semibold text-gray-800 mb-4">Confidence Score</h3>
      
      <div className="relative">
        <ResponsiveContainer width="100%" height={200}>
          <PieChart>
            <Pie
              data={data}
              cx="50%"
              cy="50%"
              startAngle={180}
              endAngle={0}
              innerRadius={60}
              outerRadius={80}
              dataKey="value"
            >
              <Cell fill={getColor()} />
              <Cell fill="#e5e7eb" />
            </Pie>
          </PieChart>
        </ResponsiveContainer>
        
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="text-center">
            <div className="text-4xl font-bold" style={{ color: getColor() }}>
              {percentage.toFixed(0)}%
            </div>
            <div className="text-sm text-gray-500">Confidence</div>
          </div>
        </div>
      </div>

      <div className="mt-4 space-y-2 text-sm">
        <div className="flex justify-between">
          <span className="text-gray-600">Path Quality:</span>
          <span className="font-semibold">{confidence > 0.7 ? 'Excellent' : confidence > 0.4 ? 'Good' : 'Fair'}</span>
        </div>
        <div className="flex justify-between">
          <span className="text-gray-600">Reliability:</span>
          <span className="font-semibold">{confidence > 0.7 ? 'High' : confidence > 0.4 ? 'Medium' : 'Low'}</span>
        </div>
      </div>
    </div>
  );
};