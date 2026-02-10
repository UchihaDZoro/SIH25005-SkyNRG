const AboutImpact = () => {
  const impacts = [
    { metric: 'Objective Evaluation', value: 'Eliminating 35% inter-observer variability' },
    { metric: 'Processing Speed', value: 'Under 1 minute vs 25-30 minutes manual' },
    { metric: 'Breed Coverage', value: '43 cattle + 13 buffalo breeds' },
    { metric: 'Accessibility', value: 'Remote area evaluation capability' }
  ];

  return (
    <section className="bg-white py-24">
      <div className="max-w-6xl mx-auto px-6">
        <div className="mb-16">
          <div className="text-sm font-medium tracking-widest uppercase mb-6 text-gray-400">
            Impact & Reach
          </div>
          <h2 className="text-4xl font-light">Supporting National Conservation</h2>
        </div>

        <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-12">
          {impacts.map((item, index) => (
            <div key={index} className="border-t-2 border-black pt-6">
              <div className="text-sm font-medium tracking-widest uppercase mb-3 text-gray-400">
                {item.metric}
              </div>
              <div className="text-xl font-light">{item.value}</div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default AboutImpact;

