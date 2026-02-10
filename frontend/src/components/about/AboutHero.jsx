const AboutHero = () => {
  return (
    <section className="bg-gradient-to-br from-green-900 via-green-800 to-emerald-900 text-white py-32 relative overflow-hidden">
      <div className="absolute inset-0 opacity-10">
        <div className="absolute top-20 right-10 w-96 h-96 bg-blue-500 rounded-full blur-3xl animate-pulse"></div>
      </div>
      <div className="max-w-6xl mx-auto px-6">
        <div className="max-w-4xl">
          <div className="text-sm font-medium tracking-widest uppercase mb-6 text-gray-400">
            Smart India Hackathon 2025
          </div>
          <h1 className="text-6xl md:text-7xl font-light mb-8 leading-tight">
            AI-Powered Livestock Intelligence for Indigenous Breed Conservation
          </h1>
          <p className="text-xl text-gray-400 leading-relaxed">
            Developing computer vision solutions to standardize livestock evaluation, 
            support scientific breeding programs, and preserve India's genetic heritage.
          </p>
        </div>
      </div>
    </section>
  );
};

export default AboutHero;
