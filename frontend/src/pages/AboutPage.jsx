import AboutHero from '../components/about/AboutHero';
import AboutMission from '../components/about/AboutMission';
import AboutApproach from '../components/about/AboutApproach';
import AboutImpact from '../components/about/AboutImpact';

const AboutPage = () => {
  return (
    <div className="min-h-screen bg-white">
      <AboutHero />
      <AboutMission />
      <AboutApproach />
      <AboutImpact />
      
      {/* Tech Stack */}
      <section className="bg-black text-white py-24">
        <div className="max-w-6xl mx-auto px-6">
          <div className="text-center mb-16">
            <div className="text-sm font-medium tracking-widest uppercase mb-6 text-gray-400">
              Technology Stack
            </div>
            <h2 className="text-4xl font-light">Built with Modern Tools</h2>
          </div>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
            {['YOLOv8', 'React', 'FastAPI', 'Python'].map((tech) => (
              <div key={tech} className="text-center py-12 border border-white/10 hover:border-white/30 transition-colors">
                <div className="text-2xl font-light">{tech}</div>
              </div>
            ))}
          </div>
        </div>
      </section>
    </div>
  );
};

export default AboutPage;
