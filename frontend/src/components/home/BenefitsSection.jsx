import { FiUsers, FiUserCheck, FiTrendingUp } from 'react-icons/fi';

const BenefitsSection = () => {
  const stakeholders = [
    {
      icon: FiTrendingUp,
      title: 'For Government Bodies',
      benefits: [
        'Standardized, objective evaluation data',
        'Reduced dependency on manual evaluators',
        'Scalable solution for national programs',
        'Digital database for policy decisions',
        'Cost-effective livestock management'
      ]
    },
    {
      icon: FiUserCheck,
      title: 'For Field Officers',
      benefits: [
        'User-friendly mobile interface',
        'Rapid evaluation process',
        'Consistent scoring methodology',
        'Digital record keeping',
        'Simplified training requirements'
      ]
    },
    {
      icon: FiUsers,
      title: 'For Livestock Owners',
      benefits: [
        'Objective animal assessment',
        'Digital certificates & records',
        'Data-driven breeding decisions',
        'Transparent evaluation process',
        'Easy access to animal history'
      ]
    }
  ];

  return (
    <section className="py-20 lg:py-32 bg-gray-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Section Header */}
        <div className="text-center mb-20 space-y-6">
          <h2 className="text-4xl sm:text-5xl lg:text-6xl font-bold text-black">
            Transforming the Ecosystem
          </h2>
          <p className="text-xl text-gray-600 max-w-3xl mx-auto">
            Creating value for every stakeholder in livestock breeding programs
          </p>
        </div>

        {/* Stakeholder Cards */}
        <div className="grid md:grid-cols-3 gap-10 mb-20">
          {stakeholders.map((stakeholder, index) => {
            const Icon = stakeholder.icon;
            return (
              <div key={index} className="relative overflow-hidden group hover:shadow-2xl transition-all bg-white rounded-3xl p-10 border-2 border-gray-200 hover:border-orange-500">
                {/* Gradient Background */}
                <div className="absolute top-0 right-0 w-32 h-32 bg-orange-100 opacity-20 rounded-full -mr-16 -mt-16 group-hover:scale-150 transition-transform duration-500"></div>

                <div className="relative space-y-6">
                  {/* Icon */}
                  <div className={`inline-flex p-5 rounded-2xl shadow-lg ${
                    index === 0 ? 'bg-gradient-to-br from-green-500 to-green-600' :
                    index === 1 ? 'bg-gradient-to-br from-blue-500 to-blue-600' :
                    'bg-gradient-to-br from-orange-500 to-orange-600'
                  }`}>
                    <Icon className="w-8 h-8 text-white" />
                  </div>

                  {/* Title */}
                  <h3 className="text-2xl font-bold text-black">
                    {stakeholder.title}
                  </h3>

                  {/* Benefits List */}
                  <ul className="space-y-4">
                    {stakeholder.benefits.map((benefit, idx) => (
                      <li key={idx} className="flex items-start space-x-3">
                        <svg className="w-5 h-5 text-green-500 mt-0.5 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                          <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                        </svg>
                        <span className="text-gray-600">{benefit}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              </div>
            );
          })}
        </div>

        {/* Impact Statement */}
        <div className="bg-gradient-to-br from-orange-500 via-orange-600 to-orange-700 text-white rounded-3xl shadow-2xl p-10 lg:p-16">
          <h3 className="text-3xl font-bold mb-6 text-center">
            Supporting National Breeding Programs
          </h3>
          <div className="prose prose-lg prose-invert max-w-none text-center">
            <p className="text-white/90 leading-relaxed text-lg">
              This solution revolutionizes livestock type classification across India, 
              supporting the Rashtriya Gokul Mission's goal of conserving and developing indigenous cattle breeds. 
              By bridging traditional evaluation methods with modern AI technology, 
              it makes scientific breeding accessible to remote areas and improves data quality for national breeding programs.
            </p>
          </div>
        </div>
      </div>
    </section>
  );
};

export default BenefitsSection;
