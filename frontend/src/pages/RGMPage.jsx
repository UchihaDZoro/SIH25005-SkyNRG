import RGMHero from '../components/rgm/RGMHero';
import RGMObjectives from '../components/rgm/RGMObjectives';
import RGMComponents from '../components/rgm/RGMComponents';

const RGMPage = () => {
  return (
    <div className="min-h-screen bg-white">
      <RGMHero />
      <RGMObjectives />
      <RGMComponents />
      
      {/* Stats */}
      <section className="bg-white py-24">
        <div className="max-w-6xl mx-auto px-6">
          <div className="grid md:grid-cols-3 gap-16">
            {[
              { stat: '43', label: 'Indigenous Cattle Breeds' },
              { stat: '13', label: 'Indigenous Buffalo Breeds' },
              { stat: '100+', label: 'Gokul Gram Centers' }
            ].map((item, index) => (
              <div key={index} className="text-center border-t-2 border-black pt-8">
                <div className="text-6xl font-light mb-4">{item.stat}</div>
                <div className="text-sm font-medium tracking-widest uppercase text-gray-400">
                  {item.label}
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="bg-black text-white py-24">
        <div className="max-w-4xl mx-auto px-6 text-center">
          <h2 className="text-4xl font-light mb-6">AI Supporting RGM Goals</h2>
          <p className="text-xl text-gray-400 mb-12">
            Our classification system provides standardized evaluation data 
            that supports breed conservation and genetic improvement programs
          </p>
          <a href="/classify">
            <button className="px-12 py-4 bg-white text-black font-medium tracking-wide hover:bg-gray-200 transition-colors">
              TRY CLASSIFICATION
            </button>
          </a>
        </div>
      </section>
    </div>
  );
};

export default RGMPage;
