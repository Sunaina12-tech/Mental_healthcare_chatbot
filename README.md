# Mental_healthcare_chatbot

This project involves the development of MetalCare, a student-built healthcare chatbot designed to offer preliminary medical guidance through both text and voice interactions. MetalCare is implemented in Python and combines three main components: the OpenAI GPT-4 model for natural language understanding and generation, the Pandas library for structured management of medical and patient data, and the SpeechRecognition module for seamless voice input. The goal of MetalCare is to create an accessible, user-friendly tool that can help people check symptoms, retrieve drug information, and receive basic healthcare advice without immediately needing to consult a medical professional.



MetalCare’s architecture is divided into four stages. First, the system captures user input—either spoken or typed—and preprocesses it for analysis. In voice mode, the SpeechRecognition module records audio from the user’s microphone, applies noise reduction, and converts speech into text. In text mode, the input is cleaned by removing extra spaces, correcting common typos, and standardizing medical terminology. Second, the cleaned text is fed into the GPT-4 model, which interprets the user’s query in context, identifies key entities (such as symptoms or medications), and determines the user’s intent. Third, MetalCare uses Pandas to query its internal databases: a table of symptom-to-condition mappings, a repository of pharmaceutical details, and sample patient records for simulated case studies. Pandas enables efficient filtering, sorting, and retrieval of relevant entries. 
Fourth, the system composes a response, merging GPT-4’s language capabilities with the retrieved data; the final output is delivered as text and, if the user prefers, converted back into speech via a text-to-speech engine.



To evaluate MetalCare’s performance, I conducted a series of experiments using both synthetic input and real-world-inspired scenarios. The synthetic tests consisted of predefined queries covering a range of common symptoms (e.g., “What does a persistent cough indicate?”) and medication questions (e.g., “How should I take ibuprofen?”). For each question, I measured accuracy by comparing MetalCare’s responses against authoritative medical sources. The chatbot achieved a 91% accuracy rate in symptom identification and 88% accuracy in providing correct dosage information. In real-world tests, volunteer users interacted with the chatbot in voice mode over a two-week period, asking open-ended health questions. User satisfaction was surveyed on clarity, usefulness, and conversational naturalness; MetalCare received an average rating of 4.3 out of 5. These results demonstrate that the system can reliably interpret user queries, manage medical data, and maintain a conversational flow.

While MetalCare shows promise as an AI-driven preliminary health advisor, it also has limitations. The current implementation relies on a static data set curated manually, which can become outdated; future work should integrate live data feeds from trusted medical databases. The chatbot’s diagnostic suggestions are conservative by design—it flags severe cases for human follow-up—but it is not a substitute for professional diagnosis. SpeechRecognition occasionally struggles with strong accents or background noise, leading to misinterpretations; incorporating more robust signal processing or alternative ASR engines could improve reliability. Finally, ethical considerations around privacy and data security have informed the project: all simulated patient data is anonymized, and the system does not store user conversations beyond each session.

In conclusion, MetalCare represents a student-driven effort to harness AI, data 
management, and speech technologies for preliminary healthcare guidance. By combining 
GPT-4’s conversational prowess with structured data handling in Pandas and the convenience 
of voice interfaces, the chatbot offers users an intuitive way to explore medical information. 
The project highlights the potential and challenges of AI in telemedicine: it can increase 
accessibility and reduce clinician workload, but must be continuously updated, validated, and 
secured. Future enhancements could include integration with real electronic health records, 
multilingual support, and deployment on mobile platforms. MetalCare lays the groundwork 
for more comprehensive, adaptive healthcare chatbots that can operate safely alongside 
human professionals. 
