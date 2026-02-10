const DocsPage = () => {
  const sections = [
    {
      title: 'Getting Started',
      content: [
        {
          heading: 'System Requirements',
          text: 'Smartphone with 8MP+ camera, stable internet connection, modern web browser (Chrome, Firefox, Safari, Edge)'
        },
        {
          heading: 'Quick Start Guide',
          text: '1. Navigate to classify page 2. Capture 6 standardized images 3. Upload photos 4. Review type score results 5. Download evaluation report'
        }
      ]
    },
    {
      title: 'Classification Process',
      content: [
        {
          heading: 'Image Capture Guidelines',
          text: 'All photos must be taken in good lighting with the animal standing naturally. Ensure the full body is visible from each required angle. Avoid shadows, distractions, and obstructions.'
        },
        {
          heading: 'AI Analysis',
          text: 'YOLOv8 pose estimation detects 30+ anatomical keypoints. Computer vision algorithms extract measurements including body lengths, angles, and proportions. Results are mapped to breed-specific type score standards.'
        },
        {
          heading: 'Type Score Generation',
          text: 'Measurements are evaluated against official breed standards. Scores reflect conformation quality across multiple traits. Results include detailed breakdown by body section and overall classification grade.'
        }
      ]
    },
    {
      title: 'Best Practices',
      content: [
        {
          heading: 'Photography Tips',
          text: 'Use natural daylight for consistent lighting. Position camera at animal shoulder height. Maintain 3-4 meter distance for full body capture. Ensure animal is standing square on level ground.'
        },
        {
          heading: 'Common Mistakes',
          text: 'Avoid tilted camera angles, partial body frames, poor lighting conditions, busy backgrounds, and images where the animal is moving or in unnatural positions.'
        }
      ]
    },
    {
      title: 'Understanding Results',
      content: [
        {
          heading: 'Score Interpretation',
          text: 'Type scores range from 0-100. Excellent: 85+, Very Good: 75-84, Good: 65-74, Fair: 55-64, Poor: <55. Scores reflect adherence to breed-specific conformation standards.'
        },
        {
          heading: 'Detailed Breakdown',
          text: 'Results include section scores for frame, dairy character, body capacity, feet & legs, and udder (for females). Each section contributes to the overall type classification.'
        }
      ]
    }
  ];

  return (
    <div className="min-h-screen bg-white">
      {/* Hero */}
      <section className="bg-black text-white py-32">
        <div className="max-w-6xl mx-auto px-6">
          <div className="max-w-4xl">
            <div className="text-sm font-medium tracking-widest uppercase mb-6 text-gray-400">
              Documentation
            </div>
            <h1 className="text-5xl md:text-6xl lg:text-7xl font-light mb-8 leading-tight">
              Complete User Guide
            </h1>
            <p className="text-xl text-gray-400 leading-relaxed">
              Comprehensive documentation for using the AI livestock classification system, 
              from image capture to understanding evaluation results.
            </p>
          </div>
        </div>
      </section>

      {/* Documentation Sections */}
      {sections.map((section, sectionIndex) => (
        <section
          key={sectionIndex}
          className={`py-24 ${sectionIndex % 2 === 0 ? 'bg-white' : 'bg-gray-50'}`}
        >
          <div className="max-w-5xl mx-auto px-6">
            <div className="mb-16">
              <div className="text-sm font-medium tracking-widest uppercase mb-6 text-gray-400">
                {String(sectionIndex + 1).padStart(2, '0')}
              </div>
              <h2 className="text-4xl font-light">{section.title}</h2>
            </div>

            <div className="space-y-12">
              {section.content.map((item, itemIndex) => (
                <div key={itemIndex} className="border-l-2 border-black pl-8">
                  <h3 className="text-2xl font-light mb-4">{item.heading}</h3>
                  <p className="text-gray-600 leading-relaxed text-lg">{item.text}</p>
                </div>
              ))}
            </div>
          </div>
        </section>
      ))}

      {/* CTA */}
      <section className="bg-black text-white py-24">
        <div className="max-w-4xl mx-auto px-6 text-center">
          <h2 className="text-4xl font-light mb-6">Ready to Start?</h2>
          <p className="text-xl text-gray-400 mb-12">
            Put this knowledge into practice with our classification system
          </p>
          <a href="/classify">
            <button className="px-12 py-4 bg-white text-black font-medium tracking-wide hover:bg-gray-200 transition-colors">
              START CLASSIFICATION
            </button>
          </a>
        </div>
      </section>
    </div>
  );
};

export default DocsPage;
