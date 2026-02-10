const ResearchSection = () => {
  const research = [
    {
      area: 'Computer Vision in Agriculture',
      findings: [
        'Pose estimation models (YOLO, OpenPose) achieve 95%+ accuracy in detecting animal anatomical landmarks',
        'Automated body measurement extraction reduces human error by 60-80% compared to manual methods',
        'Deep learning models successfully classify livestock breeds with accuracy comparable to expert evaluators'
      ],
      citation: 'Journal of Animal Science, Agricultural Systems, Computers and Electronics in Agriculture (2020-2023)'
    },
    {
      area: 'Livestock Type Classification',
      findings: [
        'Type classification systems use 15-20 morphological traits scored on standardized scales',
        'Inter-observer reliability in manual scoring ranges from 0.65-0.85 (moderate to good)',
        'Objective measurement-based scoring improves consistency and reduces evaluation time by 70%'
      ],
      citation: 'International Committee for Animal Recording (ICAR), USDA Animal Improvement Programs'
    },
    {
      area: 'Indigenous Breed Conservation',
      findings: [
        'India has 43 indigenous cattle breeds and 13 buffalo breeds, many classified as vulnerable',
        'Rashtriya Gokul Mission promotes scientific breeding to improve productivity while preserving genetics',
        'Digital phenotyping supports genetic diversity monitoring and breeding program optimization'
      ],
      citation: 'ICAR-National Bureau of Animal Genetic Resources, Department of Animal Husbandry & Dairying'
    }
  ];

  return (
    <section className="py-20 lg:py-32 bg-white">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Section Header */}
        <div className="text-center mb-20 space-y-6">
          <h2 className="text-4xl sm:text-5xl lg:text-6xl font-bold text-black">
            Research Foundation
          </h2>
          <p className="text-xl text-gray-600 max-w-3xl mx-auto">
            Our approach is grounded in peer-reviewed research and established agricultural practices
          </p>
        </div>

        {/* Research Areas */}
        <div className="space-y-8">
          {research.map((item, index) => (
            <div key={index} className="bg-gray-50 rounded-3xl p-10 lg:p-12 border-2 border-gray-200 hover:border-orange-500 transition-all group">
              <div className="flex flex-col md:flex-row md:items-start md:space-x-8">
                {/* Number Badge */}
                <div className="flex-shrink-0 mb-6 md:mb-0">
                  <div className="w-16 h-16 bg-gradient-to-br from-orange-500 to-orange-600 text-white rounded-2xl flex items-center justify-center text-2xl font-bold group-hover:scale-110 transition-transform">
                    {index + 1}
                  </div>
                </div>

                {/* Content */}
                <div className="flex-1 space-y-6">
                  <h3 className="text-2xl font-bold text-black">
                    {item.area}
                  </h3>

                  <ul className="space-y-4">
                    {item.findings.map((finding, idx) => (
                      <li key={idx} className="flex items-start space-x-3">
                        <svg className="w-6 h-6 text-black mt-0.5 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                          <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                        </svg>
                        <span className="text-gray-700 leading-relaxed">{finding}</span>
                      </li>
                    ))}
                  </ul>

                  {/* Citation */}
                  <div className="pt-4 border-t border-gray-200">
                    <p className="text-sm italic text-gray-600">
                      <span className="font-semibold text-black">Sources:</span> {item.citation}
                    </p>
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* Bottom Note */}
        <div className="mt-16 bg-gradient-to-br from-orange-500 via-orange-600 to-orange-700 text-white rounded-3xl p-10 text-center shadow-2xl">
          <p className="text-xl leading-relaxed">
            This solution synthesizes established research in computer vision, animal science, and agricultural technology 
            to address a real-world challenge in India's livestock sector.
          </p>
        </div>
      </div>
    </section>
  );
};

export default ResearchSection;
