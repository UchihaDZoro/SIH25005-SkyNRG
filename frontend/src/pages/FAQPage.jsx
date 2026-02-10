import FAQHero from '../components/faq/FAQHero';
import FAQList from '../components/faq/FAQList';

const FAQPage = () => {
  return (
    <div className="min-h-screen bg-white">
      <FAQHero />
      <FAQList />
      
      {/* CTA */}
      <section className="bg-black text-white py-24">
        <div className="max-w-4xl mx-auto px-6 text-center">
          <h2 className="text-4xl font-light mb-6">Still Have Questions?</h2>
          <p className="text-xl text-gray-400 mb-12">
            Our team is available to provide detailed technical assistance
          </p>
          <a href="/contact">
            <button className="px-12 py-4 bg-white text-black font-medium tracking-wide hover:bg-gray-200 transition-colors">
              CONTACT SUPPORT
            </button>
          </a>
        </div>
      </section>
    </div>
  );
};

export default FAQPage;
