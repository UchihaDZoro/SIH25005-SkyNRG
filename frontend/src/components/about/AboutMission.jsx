const AboutMission = () => {
  return (
    <section className="bg-white py-24">
      <div className="max-w-6xl mx-auto px-6">
        <div className="grid md:grid-cols-2 gap-20">
          {/* Mission */}
          <div>
            <div className="text-sm font-medium tracking-widest uppercase mb-8 text-gray-400">
              01 — Mission
            </div>
            <h2 className="text-4xl font-light mb-8 leading-tight">
              Democratizing Scientific Livestock Evaluation
            </h2>
            <p className="text-gray-600 leading-relaxed text-lg">
              Our mission is to develop an accessible AI-powered classification system 
              that provides objective, standardized livestock evaluation data. By leveraging 
              computer vision and pose estimation, we enable data-driven breeding decisions 
              that support indigenous breed conservation across India.
            </p>
          </div>

          {/* Vision */}
          <div>
            <div className="text-sm font-medium tracking-widest uppercase mb-8 text-gray-400">
              02 — Vision
            </div>
            <h2 className="text-4xl font-light mb-8 leading-tight">
              Preserving Genetic Heritage Through Technology
            </h2>
            <p className="text-gray-600 leading-relaxed text-lg">
              We envision a future where every livestock owner, from remote villages to 
              government breeding centers, has access to world-class AI evaluation tools. 
              This technology will preserve India's 43 cattle and 13 buffalo breeds while 
              improving productivity and supporting national conservation goals.
            </p>
          </div>
        </div>
      </div>
    </section>
  );
};

export default AboutMission;
