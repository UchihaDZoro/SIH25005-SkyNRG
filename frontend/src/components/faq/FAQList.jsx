import { useState } from 'react';

const FAQList = () => {
  const [openIndex, setOpenIndex] = useState(null);

  const faqs = [
    {
      category: 'System Overview',
      questions: [
        {
          q: 'What is the Animal Classification System?',
          a: 'An AI-powered solution using YOLOv8 pose estimation to automatically evaluate livestock based on morphological traits, generating objective type scores according to breed standards.'
        },
        {
          q: 'How does it support the Rashtriya Gokul Mission?',
          a: 'Provides standardized evaluation data for indigenous breed conservation, reduces manual assessment time, and makes scientific breeding accessible to remote areas.'
        },
        {
          q: 'What animals can be classified?',
          a: 'Indigenous Indian cattle and buffalo breeds, evaluating body structure, udder conformation, leg placement, and breed-specific characteristics.'
        }
      ]
    },
    {
      category: 'Technical Details',
      questions: [
        {
          q: 'What technology powers the classification?',
          a: 'YOLOv8 detects 30+ anatomical keypoints which are processed to extract body measurements (lengths, angles, ratios) mapped to breed-specific type scores.'
        },
        {
          q: 'What is the classification accuracy?',
          a: 'AI systems achieve 95%+ accuracy in keypoint detection and significantly reduce inter-observer variability compared to manual scoring (typically 35%).'
        },
        {
          q: 'What images are required?',
          a: 'Six standardized photos: front view, rear view, two lateral views, and two additional angles. Images must be clear, well-lit, showing the full animal standing.'
        }
      ]
    },
    {
      category: 'Implementation',
      questions: [
        {
          q: 'Who can use this system?',
          a: 'Government breeding programs, field officers, veterinarians, livestock owners, and anyone involved in cattle/buffalo evaluation and breeding decisions.'
        },
        {
          q: 'What equipment is needed?',
          a: 'A smartphone with a decent camera (8MP+) is sufficient to capture the required standardized photos.'
        },
        {
          q: 'How long does evaluation take?',
          a: 'AI processing completes in under 1 minute to generate a comprehensive type score report, compared to 25-30 minutes for manual evaluation.'
        }
      ]
    },
    {
      category: 'Data & Privacy',
      questions: [
        {
          q: 'How is data used?',
          a: 'All data is used solely for livestock evaluation. With permission, anonymized data may improve the AI model and support national breeding analytics.'
        },
        {
          q: 'Is information secure?',
          a: 'Industry-standard security with encrypted transmission, secure storage, and access controls. Personal and animal data is treated as confidential.'
        },
        {
          q: 'Can results be exported?',
          a: 'Yes. Downloadable PDF reports with detailed scores and measurements. Data can be exported in structured formats for database integration.'
        }
      ]
    }
  ];

  return (
    <section className="bg-white py-24">
      <div className="max-w-4xl mx-auto px-6">
        {faqs.map((category, catIndex) => (
          <div key={catIndex} className="mb-20">
            <h2 className="text-sm font-medium tracking-widest uppercase mb-12 text-gray-400">
              {category.category}
            </h2>
            
            <div className="space-y-1">
              {category.questions.map((faq, qIndex) => {
                const key = `${catIndex}-${qIndex}`;
                const isOpen = openIndex === key;
                
                return (
                  <div key={qIndex} className="border-b border-gray-200">
                    <button
                      onClick={() => setOpenIndex(isOpen ? null : key)}
                      className="w-full py-6 flex items-start justify-between text-left hover:bg-gray-50 transition-colors px-4"
                    >
                      <span className="text-xl font-light pr-8 leading-relaxed">
                        {faq.q}
                      </span>
                      <span className="text-2xl font-light flex-shrink-0 transition-transform" style={{
                        transform: isOpen ? 'rotate(45deg)' : 'rotate(0deg)'
                      }}>
                        +
                      </span>
                    </button>
                    
                    <div className={`overflow-hidden transition-all duration-300 ${
                      isOpen ? 'max-h-96 pb-6' : 'max-h-0'
                    }`}>
                      <div className="px-4 text-gray-600 leading-relaxed">
                        {faq.a}
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        ))}
      </div>
    </section>
  );
};

export default FAQList;
