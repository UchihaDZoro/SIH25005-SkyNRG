const ContactInfo = () => {
  return (
    <section className="bg-white py-24">
      <div className="max-w-6xl mx-auto px-6">
        <div className="grid md:grid-cols-2 gap-20">
          {/* Left - Why Contact */}
          <div>
            <h2 className="text-4xl font-light mb-8">Why Partner With Us</h2>
            <div className="space-y-8">
              <div className="border-l-2 border-black pl-6">
                <h3 className="text-xl font-medium mb-3">Research Collaboration</h3>
                <p className="text-gray-600 leading-relaxed">
                  Work with our team on advancing computer vision applications 
                  in livestock science and breed conservation.
                </p>
              </div>
              
              <div className="border-l-2 border-black pl-6">
                <h3 className="text-xl font-medium mb-3">Implementation Support</h3>
                <p className="text-gray-600 leading-relaxed">
                  Get technical assistance deploying the classification system 
                  in your breeding programs or government initiatives.
                </p>
              </div>
              
              <div className="border-l-2 border-black pl-6">
                <h3 className="text-xl font-medium mb-3">Data Partnership</h3>
                <p className="text-gray-600 leading-relaxed">
                  Contribute anonymized evaluation data to improve model accuracy 
                  and support national breed conservation goals.
                </p>
              </div>
            </div>
          </div>

          {/* Right - Contact Details */}
          <div>
            <h2 className="text-4xl font-light mb-8">Reach Our Team</h2>
            <div className="space-y-8">
              <div>
                <div className="text-sm font-medium tracking-widest uppercase mb-2 text-gray-400">
                  General Inquiries
                </div>
                <a href="mailto:info@animalclassification.com" className="text-2xl font-light hover:text-gray-600 transition-colors">
                  info@animalclassification.com
                </a>
              </div>

              <div>
                <div className="text-sm font-medium tracking-widest uppercase mb-2 text-gray-400">
                  Technical Support
                </div>
                <a href="mailto:support@animalclassification.com" className="text-2xl font-light hover:text-gray-600 transition-colors">
                  support@animalclassification.com
                </a>
              </div>

              <div>
                <div className="text-sm font-medium tracking-widest uppercase mb-2 text-gray-400">
                  Phone
                </div>
                <a href="tel:+911234567890" className="text-2xl font-light hover:text-gray-600 transition-colors">
                  +91 12345 67890
                </a>
              </div>

              <div>
                <div className="text-sm font-medium tracking-widest uppercase mb-2 text-gray-400">
                  Location
                </div>
                <p className="text-2xl font-light text-gray-700">
                  New Delhi, India
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default ContactInfo;
