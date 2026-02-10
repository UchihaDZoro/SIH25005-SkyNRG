const ContactHero = () => {
  return (
    <section className="bg-gradient-to-br from-purple-900 via-purple-800 to-indigo-900 text-white py-32 relative overflow-hidden">
      <div className="absolute inset-0 opacity-10">
        <div className="absolute bottom-20 left-10 w-96 h-96 bg-orange-500 rounded-full blur-3xl animate-pulse"></div>
      </div>
      <div className="max-w-6xl mx-auto px-6">
        <div className="max-w-3xl">
          <div className="text-sm font-medium tracking-widest uppercase mb-6 text-gray-400">
            Get in Touch
          </div>
          <h1 className="text-6xl md:text-7xl font-light mb-8 leading-tight">
            Let's Build the Future of Livestock Evaluation
          </h1>
          <p className="text-xl text-gray-400 leading-relaxed">
            Connect with our team to discuss how AI-powered classification 
            can transform your breeding programs and conservation efforts.
          </p>
        </div>
      </div>
    </section>
  );
};

export default ContactHero;
