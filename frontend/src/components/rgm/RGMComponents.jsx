const RGMComponents = () => {
  const components = [
    {
      title: 'Breed Development Centers',
      desc: 'Scientific breeding facilities for genetic improvement and germplasm production'
    },
    {
      title: 'Bull Production',
      desc: 'Elite indigenous bull production and distribution to farmer communities'
    },
    {
      title: 'Capacity Building',
      desc: 'Training programs for farmers, field officers, and breeding professionals'
    },
    {
      title: 'E-Pashuhaat Portal',
      desc: 'Digital marketplace connecting breeders, buyers, and germplasm resources'
    },
    {
      title: 'Breed Registration',
      desc: 'Comprehensive documentation and registration of indigenous animals'
    },
    {
      title: 'Research & Development',
      desc: 'Scientific studies on nutrition, genetics, and breed characteristics'
    }
  ];

  return (
    <section className="bg-gray-50 py-24">
      <div className="max-w-6xl mx-auto px-6">
        <div className="mb-16">
          <div className="text-sm font-medium tracking-widest uppercase mb-6 text-gray-400">
            Program Components
          </div>
          <h2 className="text-4xl font-light">Implementation Framework</h2>
        </div>

        <div className="grid md:grid-cols-3 gap-12">
          {components.map((comp, index) => (
            <div key={index} className="border-t-2 border-black pt-6">
              <h3 className="text-xl font-light mb-3">{comp.title}</h3>
              <p className="text-gray-600 leading-relaxed text-sm">{comp.desc}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default RGMComponents;

