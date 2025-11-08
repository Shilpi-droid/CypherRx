// src/components/AnswerDisplay.tsx
import React from 'react';
import { CheckCircle2, AlertCircle } from 'lucide-react';

interface AnswerDisplayProps {
  answer: string;
  confidence: number;
}

export const AnswerDisplay: React.FC<AnswerDisplayProps> = ({ answer, confidence }) => {
  const getConfidenceColor = () => {
    if (confidence > 0.7) return 'text-green-600';
    if (confidence > 0.4) return 'text-yellow-600';
    return 'text-red-600';
  };

  const getConfidenceIcon = () => {
    if (confidence > 0.7) return <CheckCircle2 className="w-5 h-5" />;
    return <AlertCircle className="w-5 h-5" />;
  };

  return (
    <div className="bg-white rounded-2xl shadow-xl p-6 mb-6">
      <div className="flex items-start justify-between mb-4">
        <h2 className="text-2xl font-bold text-gray-800">Answer</h2>
        <div className={`flex items-center gap-2 ${getConfidenceColor()} font-semibold`}>
          {getConfidenceIcon()}
          <span>{(confidence * 100).toFixed(0)}% Confident</span>
        </div>
      </div>
      
      <p className="text-lg text-gray-700 leading-relaxed">
        {answer}
      </p>

      {confidence < 0.7 && (
        <div className="mt-4 p-4 bg-yellow-50 border-l-4 border-yellow-400 rounded">
          <p className="text-sm text-yellow-800">
            <strong>Note:</strong> This answer has moderate confidence. Consider consulting additional sources.
          </p>
        </div>
      )}
    </div>
  );
};