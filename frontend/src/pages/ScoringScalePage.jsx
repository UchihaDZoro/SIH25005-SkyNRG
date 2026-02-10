import React from 'react';
import { FiInfo } from 'react-icons/fi';

const ScoringScalePage = () => {
  const scoringData = {
    dairyStrength: [
      { score: 1, stature: '≤110', heartGirth: '≤145', bodyLength: '≤115', bodyDepth: '≤58', angularity: 'Coarse' },
      { score: 2, stature: '111-113', heartGirth: '146-149', bodyLength: '116-118', bodyDepth: '59', angularity: 'Slightly coarse' },
      { score: 3, stature: '114-116', heartGirth: '150-153', bodyLength: '119-121', bodyDepth: '60-61', angularity: 'Moderate' },
      { score: 4, stature: '117-118', heartGirth: '154-157', bodyLength: '122-123', bodyDepth: '62', angularity: 'Slightly angular' },
      { score: 5, stature: '119-121', heartGirth: '158-162', bodyLength: '124-126', bodyDepth: '63-64', angularity: 'Intermediate' },
      { score: 6, stature: '122-123', heartGirth: '163-165', bodyLength: '127-128', bodyDepth: '65', angularity: 'Moderately angular' },
      { score: 7, stature: '124-125', heartGirth: '166-168', bodyLength: '129-131', bodyDepth: '66-67', angularity: 'Angular' },
      { score: 8, stature: '126-127', heartGirth: '169-171', bodyLength: '132-134', bodyDepth: '68-69', angularity: 'Very angular' },
      { score: 9, stature: '≥128', heartGirth: '≥172', bodyLength: '≥135', bodyDepth: '≥70', angularity: 'Extremely angular' }
    ],
    rump: [
      { score: 1, angle: 'High pins (>12 cm)', width: '≤12' },
      { score: 2, angle: '11-12 cm', width: '13' },
      { score: 3, angle: '10-11 cm', width: '14-15' },
      { score: 4, angle: '9-10 cm', width: '16' },
      { score: 5, angle: '8-9 cm', width: '17-19' },
      { score: 6, angle: '7-8 cm', width: '20' },
      { score: 7, angle: '6-7 cm', width: '21-22' },
      { score: 8, angle: '5-6 cm', width: '23' },
      { score: 9, angle: 'Very low pins (<5 cm)', width: '≥24' }
    ],
    feetLegs: [
      { score: 1, rearLegsSet: 'Straight (170°)', rearLegsView: 'Toe out', footAngle: '<42' },
      { score: 2, rearLegsSet: '165-169°', rearLegsView: 'Slight toe out', footAngle: '43-44' },
      { score: 3, rearLegsSet: '160-164°', rearLegsView: 'Moderate deviation', footAngle: '45-46' },
      { score: 4, rearLegsSet: '156-159°', rearLegsView: 'Nearly parallel', footAngle: '47' },
      { score: 5, rearLegsSet: '150-155°', rearLegsView: 'Parallel', footAngle: '46-50°' },
      { score: 6, rearLegsSet: '146-149°', rearLegsView: 'Slight bow', footAngle: '51' },
      { score: 7, rearLegsSet: '141-145°', rearLegsView: 'Moderate bow', footAngle: '52-55' },
      { score: 8, rearLegsSet: '135-140°', rearLegsView: 'Pronounced bow', footAngle: '56-59' },
      { score: 9, rearLegsSet: 'Sickled (≤134°)', rearLegsView: 'Severe bow-legged', footAngle: '≥60' }
    ],
    udder: [
      { score: 1, foreUdder: 'Loose', rearHeight: '<8', centralLigament: 'Weak', udderDepth: '>10 below' },
      { score: 2, foreUdder: 'Weak', rearHeight: '8-10', centralLigament: 'Very weak', udderDepth: '8-10 below' },
      { score: 3, foreUdder: 'Moderate', rearHeight: '10-12', centralLigament: 'Shallow', udderDepth: '5-8 below' },
      { score: 4, foreUdder: 'Good', rearHeight: '12-14', centralLigament: 'Moderate', udderDepth: '2-5 below' },
      { score: 5, foreUdder: 'Smooth/tight', rearHeight: '14-16', centralLigament: 'Deep cleft', udderDepth: 'At hock level' },
      { score: 6, foreUdder: 'Strong', rearHeight: '16-18', centralLigament: 'Strong', udderDepth: 'At hock' },
      { score: 7, foreUdder: 'Very strong', rearHeight: '18-20', centralLigament: 'Very strong', udderDepth: '2-5 above' },
      { score: 8, foreUdder: 'Extremely strong', rearHeight: '>20', centralLigament: 'Extremely strong', udderDepth: '5-8 above' },
      { score: 9, foreUdder: 'Board-like', rearHeight: '>22', centralLigament: 'Overly defined', udderDepth: '>10 above' }
    ]
  };

  return (
    <div className="min-h-screen bg-gray-50 pt-20">
      {/* Header */}
      <div className="bg-gradient-to-r from-slate-900 to-slate-800 text-white py-12 border-b-4 border-orange-500">
        <div className="max-w-7xl mx-auto px-6">
          <div className="flex items-center gap-4">
            <FiInfo className="w-12 h-12" />
            <div>
              <p className="text-xs font-bold tracking-widest uppercase text-gray-400 mb-2">Reference Guide</p>
              <h1 className="text-4xl font-bold uppercase tracking-tight">Gir Cattle Scoring Scale</h1>
              <p className="text-gray-300 mt-2">Official trait scoring criteria for classification</p>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-6 py-12">
        {/* Dairy Strength */}
        <div className="bg-white border-2 border-gray-200 overflow-hidden mb-8">
          <div className="bg-slate-900 text-white px-8 py-6 border-b-4 border-orange-500">
            <h2 className="text-2xl font-bold uppercase tracking-wide">A) Dairy Strength</h2>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full min-w-max">
              <thead className="bg-gray-100">
                <tr>
                  <th className="px-6 py-4 text-left text-xs font-bold uppercase text-gray-700 border-b-2 border-gray-300">Score</th>
                  <th className="px-6 py-4 text-left text-xs font-bold uppercase text-gray-700 border-b-2 border-gray-300">Stature (cm)</th>
                  <th className="px-6 py-4 text-left text-xs font-bold uppercase text-gray-700 border-b-2 border-gray-300">Heart Girth (cm)</th>
                  <th className="px-6 py-4 text-left text-xs font-bold uppercase text-gray-700 border-b-2 border-gray-300">Body Length (cm)</th>
                  <th className="px-6 py-4 text-left text-xs font-bold uppercase text-gray-700 border-b-2 border-gray-300">Body Depth (cm)</th>
                  <th className="px-6 py-4 text-left text-xs font-bold uppercase text-gray-700 border-b-2 border-gray-300">Angularity</th>
                </tr>
              </thead>
              <tbody>
                {scoringData.dairyStrength.map((row, idx) => (
                  <tr key={row.score} className={idx % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
                    <td className="px-6 py-4 border-b border-gray-200 font-bold text-orange-600">{row.score}</td>
                    <td className="px-6 py-4 border-b border-gray-200">{row.stature}</td>
                    <td className="px-6 py-4 border-b border-gray-200">{row.heartGirth}</td>
                    <td className="px-6 py-4 border-b border-gray-200">{row.bodyLength}</td>
                    <td className="px-6 py-4 border-b border-gray-200">{row.bodyDepth}</td>
                    <td className="px-6 py-4 border-b border-gray-200">{row.angularity}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Rump */}
        <div className="bg-white border-2 border-gray-200 overflow-hidden mb-8">
          <div className="bg-slate-900 text-white px-8 py-6 border-b-4 border-orange-500">
            <h2 className="text-2xl font-bold uppercase tracking-wide">B) Rump</h2>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="bg-gray-100">
                <tr>
                  <th className="px-6 py-4 text-left text-xs font-bold uppercase text-gray-700 border-b-2 border-gray-300">Score</th>
                  <th className="px-6 py-4 text-left text-xs font-bold uppercase text-gray-700 border-b-2 border-gray-300">Rump Angle (hooks→pins, cm)</th>
                  <th className="px-6 py-4 text-left text-xs font-bold uppercase text-gray-700 border-b-2 border-gray-300">Rump Width (pin-pin, cm)</th>
                </tr>
              </thead>
              <tbody>
                {scoringData.rump.map((row, idx) => (
                  <tr key={row.score} className={idx % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
                    <td className="px-6 py-4 border-b border-gray-200 font-bold text-orange-600">{row.score}</td>
                    <td className="px-6 py-4 border-b border-gray-200">{row.angle}</td>
                    <td className="px-6 py-4 border-b border-gray-200">{row.width}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Feet & Legs */}
        <div className="bg-white border-2 border-gray-200 overflow-hidden mb-8">
          <div className="bg-slate-900 text-white px-8 py-6 border-b-4 border-orange-500">
            <h2 className="text-2xl font-bold uppercase tracking-wide">C) Feet & Legs</h2>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full min-w-max">
              <thead className="bg-gray-100">
                <tr>
                  <th className="px-6 py-4 text-left text-xs font-bold uppercase text-gray-700 border-b-2 border-gray-300">Score</th>
                  <th className="px-6 py-4 text-left text-xs font-bold uppercase text-gray-700 border-b-2 border-gray-300">Rear Legs Set (°)</th>
                  <th className="px-6 py-4 text-left text-xs font-bold uppercase text-gray-700 border-b-2 border-gray-300">Rear Legs Rear View</th>
                  <th className="px-6 py-4 text-left text-xs font-bold uppercase text-gray-700 border-b-2 border-gray-300">Foot Angle (°)</th>
                </tr>
              </thead>
              <tbody>
                {scoringData.feetLegs.map((row, idx) => (
                  <tr key={row.score} className={idx % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
                    <td className="px-6 py-4 border-b border-gray-200 font-bold text-orange-600">{row.score}</td>
                    <td className="px-6 py-4 border-b border-gray-200">{row.rearLegsSet}</td>
                    <td className="px-6 py-4 border-b border-gray-200">{row.rearLegsView}</td>
                    <td className="px-6 py-4 border-b border-gray-200">{row.footAngle}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Udder */}
        <div className="bg-white border-2 border-gray-200 overflow-hidden mb-8">
          <div className="bg-slate-900 text-white px-8 py-6 border-b-4 border-orange-500">
            <h2 className="text-2xl font-bold uppercase tracking-wide">D) Udder</h2>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full min-w-max">
              <thead className="bg-gray-100">
                <tr>
                  <th className="px-6 py-4 text-left text-xs font-bold uppercase text-gray-700 border-b-2 border-gray-300">Score</th>
                  <th className="px-6 py-4 text-left text-xs font-bold uppercase text-gray-700 border-b-2 border-gray-300">Fore Udder Attachment</th>
                  <th className="px-6 py-4 text-left text-xs font-bold uppercase text-gray-700 border-b-2 border-gray-300">Rear Udder Height (cm)</th>
                  <th className="px-6 py-4 text-left text-xs font-bold uppercase text-gray-700 border-b-2 border-gray-300">Central Ligament</th>
                  <th className="px-6 py-4 text-left text-xs font-bold uppercase text-gray-700 border-b-2 border-gray-300">Udder Depth</th>
                </tr>
              </thead>
              <tbody>
                {scoringData.udder.map((row, idx) => (
                  <tr key={row.score} className={idx % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
                    <td className="px-6 py-4 border-b border-gray-200 font-bold text-orange-600">{row.score}</td>
                    <td className="px-6 py-4 border-b border-gray-200">{row.foreUdder}</td>
                    <td className="px-6 py-4 border-b border-gray-200">{row.rearHeight}</td>
                    <td className="px-6 py-4 border-b border-gray-200">{row.centralLigament}</td>
                    <td className="px-6 py-4 border-b border-gray-200">{row.udderDepth}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Note */}
        <div className="bg-blue-50 border-l-4 border-blue-500 p-6">
          <p className="text-sm text-blue-900">
            <span className="font-bold">Note:</span> Scores range from 1 (lowest) to 9 (highest). Score 5 typically represents the intermediate/ideal standard for Gir cattle. Use this scale as a reference when viewing classification results.
          </p>
        </div>
      </div>
    </div>
  );
};

export default ScoringScalePage;
