const RGMObjectives = () => {
  const objectives = [
    {
      num: '01',
      title: 'Indigenous Breed Development',
      desc: 'Enhance productivity through selective breeding programs while maintaining breed purity and genetic characteristics.'
    },
    {
      num: '02',
      title: 'Genetic Conservation',
      desc: 'Preserve 43 indigenous cattle and 13 buffalo breeds from extinction and genetic erosion through scientific documentation.'
    },
    {
      num: '03',
      title: 'Infrastructure Development',
      desc: 'Establish Gokul Gram integrated cattle development centers for breed improvement and farmer capacity building.'
    },
    {
      num: '04',
      title: 'Productivity Enhancement',
      desc: 'Increase milk productivity and breeding efficiency through genetic improvement and evidence-based management.'
    }
  ];

  return (
    <section className="bg-white py-24">
      <div className="max-w-6xl mx-auto px-6">
        <div className="mb-16">
          <div className="text-sm font-medium tracking-widest uppercase mb-6 text-gray-400">
            Program Objectives
          </div>
          <h2 className="text-4xl font-light">Strategic Focus Areas</h2>
        </div>

        <div className="grid md:grid-cols-2 gap-16">
          {objectives.map((obj) => (
            <div key={obj.num} className="border-l-4 border-blue-500 hover:border-blue-600 transition-colors pl-8">
              <div className="text-sm font-medium tracking-widest uppercase mb-4 text-gray-400">
                {obj.num}
              </div>
              <h3 className="text-2xl font-light mb-4">{obj.title}</h3>
              <p className="text-gray-600 leading-relaxed">{obj.desc}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default RGMObjectives;

