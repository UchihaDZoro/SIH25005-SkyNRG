import { useState } from 'react';

const ContactForm = () => {
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    organization: '',
    subject: '',
    message: ''
  });

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    alert('Message sent successfully');
    setFormData({ name: '', email: '', organization: '', subject: '', message: '' });
  };

  return (
    <section className="bg-gray-50 py-24">
      <div className="max-w-4xl mx-auto px-6">
        <div className="mb-12">
          <h2 className="text-4xl font-light mb-4">Send a Message</h2>
          <p className="text-gray-600">We typically respond within 24 hours</p>
        </div>

        <form onSubmit={handleSubmit} className="space-y-8">
          <div className="grid md:grid-cols-2 gap-8">
            <div>
              <label className="block text-xs font-medium tracking-widest uppercase mb-3 text-gray-600">
                Name
              </label>
              <input
                type="text"
                name="name"
                value={formData.name}
                onChange={handleChange}
                required
                className="w-full px-0 py-3 bg-transparent border-0 border-b-2 border-gray-300 focus:border-black focus:outline-none transition-colors text-lg"
                placeholder="Your full name"
              />
            </div>

            <div>
              <label className="block text-xs font-medium tracking-widest uppercase mb-3 text-gray-600">
                Email
              </label>
              <input
                type="email"
                name="email"
                value={formData.email}
                onChange={handleChange}
                required
                className="w-full px-0 py-3 bg-transparent border-0 border-b-2 border-gray-300 focus:border-black focus:outline-none transition-colors text-lg"
                placeholder="your@email.com"
              />
            </div>
          </div>

          <div>
            <label className="block text-xs font-medium tracking-widest uppercase mb-3 text-gray-600">
              Organization
            </label>
            <input
              type="text"
              name="organization"
              value={formData.organization}
              onChange={handleChange}
              className="w-full px-0 py-3 bg-transparent border-0 border-b-2 border-gray-300 focus:border-black focus:outline-none transition-colors text-lg"
              placeholder="Your organization or institution"
            />
          </div>

          <div>
            <label className="block text-xs font-medium tracking-widest uppercase mb-3 text-gray-600">
              Subject
            </label>
            <input
              type="text"
              name="subject"
              value={formData.subject}
              onChange={handleChange}
              required
              className="w-full px-0 py-3 bg-transparent border-0 border-b-2 border-gray-300 focus:border-black focus:outline-none transition-colors text-lg"
              placeholder="What is this regarding?"
            />
          </div>

          <div>
            <label className="block text-xs font-medium tracking-widest uppercase mb-3 text-gray-600">
              Message
            </label>
            <textarea
              name="message"
              value={formData.message}
              onChange={handleChange}
              required
              rows="6"
              className="w-full px-0 py-3 bg-transparent border-0 border-b-2 border-gray-300 focus:border-black focus:outline-none transition-colors text-lg resize-none"
              placeholder="Tell us more about your inquiry..."
            ></textarea>
          </div>

          <div className="pt-4">
            <button
              type="submit"
              className="px-12 py-4 bg-gradient-to-r from-green-500 to-green-600 text-white font-bold tracking-wide hover:from-green-600 hover:to-green-700 transition-all rounded-lg shadow-lg hover:shadow-xl"
            >
              SEND MESSAGE
            </button>
          </div>
        </form>
      </div>
    </section>
  );
};

export default ContactForm;
