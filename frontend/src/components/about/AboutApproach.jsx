const AboutApproach = () => {
  const approaches = [
    {
      title: 'Computer Vision',
      desc: 'YOLO11m-pose estimation detects 30+ anatomical keypoints with 95%+ accuracy, eliminating human subjectivity in livestock assessment.'
    },
    {
      title: 'Standardized Methodology',
      desc: 'Consistent measurement extraction from body proportions, angles, and ratios mapped to breed-specific type score standards.'
    },
    {
      title: 'Mobile Accessibility',
      desc: 'Smartphone-based image capture enables evaluation in remote areas, requiring only basic camera equipment (8MP+).'
    },
    {
      title: 'Data Integration',
      desc: 'Structured output formats integrate with national breeding databases, supporting RGM analytics and conservation tracking.'
    }
  ];

  return (
    <section className="bg-gray-50 py-24">
      <div className="max-w-6xl mx-auto px-6">
        <div className="mb-16">
          <div className="text-sm font-medium tracking-widest uppercase mb-6 text-gray-400">
            Our Approach
          </div>
          <h2 className="text-4xl font-light">Technical Methodology</h2>
        </div>

        <div className="grid md:grid-cols-2 gap-16">
          {approaches.map((item, index) => (
            <div key={index} className="border-l-4 border-orange-500 hover:border-orange-600 transition-colors pl-8">
              <h3 className="text-2xl font-light mb-4">{item.title}</h3>
              <p className="text-gray-600 leading-relaxed">{item.desc}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default AboutApproach;

