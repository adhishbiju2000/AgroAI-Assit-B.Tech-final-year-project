# AgroAI Assist

#### AgroAI Assist: CNN-Based Leaf Disease Detection with Real-Time Weather Alerts and Multilingual Chatbot Support for Farmers in India *B.Tech Major Project | Published in Vesper Journal (Scopus Indexed)*  

## Introduction

Agriculture faces major challenges including delayed disease detection, unpredictable weather patterns, and limited expert availability—especially in rural regions. These issues cause severe crop losses and financial instability for farmers.

**AgroAI Assist** addresses these challenges by integrating:  
- **AI-powered disease detection** (EfficientNet CNN)  
- **Real-time weather forecasting**  
- **Multilingual AI chatbot**  
- **Farmer-friendly dashboard**
  
The system bridges the gap between **advanced AI research** and **practical agricultural needs**, making precision farming more accessible to rural users and makes advanced agricultural intelligence accessible to farmers, promoting early intervention and improved crop management.

<img width="800" height="600" alt="Home Page" src="https://github.com/user-attachments/assets/b04b68fb-ca1a-4c40-a3c7-ea9ced50c64f" />


## Project Leadership 
Led by **Adhish Biju**, responsible for:  
- End-to-end system development & project coordination  
- EfficientNet CNN model development & training  
- Full integration of AI modules into Python Flask  
- Research paper preparation and publication  
- Backend, frontend, deployment, and end-to-end testing

  
## Project Overview 
AgroAI Assist is an AI-driven agricultural support system that enables farmers to:  
- Detect potato leaf diseases using **EfficientNet CNN**  
- Access **real-time weather insights and five days weather forcast**  
- Communicate with a **Multilingual AI chatbot for farmers in India**  
- Make informed farming decisions through a simple web interface  


## Publication

Our research has been published in the **Vesper Journal (ISSN: 2704-7598)**—an internationally recognized, peer-reviewed journal.

### Publication Details

| **Attribute** | **Details** |
|---------------|-------------|
| **Journal** | Vesper Journal (International) |
| **ISSN** | 2704-7598 |
| **Indexing** | Scopus • Web of Science • UGC CARE Group II |
| **Category** | International Peer-Reviewed Journal |
| **Research Domain** | AI in Agriculture |
| **Type** | Original Research Article |
| **Review Process** | Double-Blind Peer Review |

<div align="center">

## Published Research Article  
**Title:** *AgroAI Assist: CNN-Based Leaf Disease Detection with Real-Time Weather Alerts and Multilingual Chatbot Support for Farmers in India*

[![View Publication](https://img.shields.io/badge/📄_View_Publication-Vesper_Journal-1a73e8?style=for-the-badge&labelColor=174ea6)](vesper09101423.pdf)
</div>


## Purpose and Objectives

### Purpose  
To equip farmers with an intelligent platform for early disease detection, weather tracking, and agricultural knowledge—empowering them to make informed decisions.

### Key Objectives  
- Accurate crop disease classification using deep learning  
- Real-time weather forecasting for planning  
- Multilingual chatbot for accessibility  
- Farmer-friendly interface  
- Demonstrate real-world application of AI/ML in agriculture  



## System Architecture

The project follows a modular architecture consisting of:

1. **Frontend UI**  
2. **Backend Flask Server**  
3. **EfficientNet CNN Model**  
4. **Weather API Integration**  
5. **Chatbot NLP Engine**  
6. **Database Layer**


<img width="638" height="298" alt="Archtechture" src="https://github.com/user-attachments/assets/75c82d7a-1d88-4e1e-b91e-3a6f9031baa9" />

---

## Algorithm Used — EfficientNet

EfficientNet-B3 was selected for its accuracy, computational efficiency, and suitability for real-time inference.

### Why EfficientNet?
- Compound scaling for balanced depth, width, and resolution  
- Superior accuracy-to-parameter ratio  
- Fast inference ideal for agricultural deployment  
- Pre-trained ImageNet weights improve leaf feature extraction  

### Implementation Specs

### Performance
- **Training Accuracy:** 96.8%  
- **Validation Accuracy:** 94.2%  
- **Inference Time:** ~180ms (CPU)  



## Application Modules

## 1. Authentication  
- Login & Signup with validation  
- JWT-based security system
<img width="800" height="600" alt="Signup" src="https://github.com/user-attachments/assets/b7ade2f2-9c43-42a0-84af-3c27ee2e1105" />

### LOGIN page after the user singup with the new account
<img width="800" height="600" alt="login" src="https://github.com/user-attachments/assets/ba47d74d-35a4-48ff-88e4-1b718d8a101b" />


## 2. Dashboard  

Central hub showing detection stats, weather widget, and quick navigation.

<img width="800" src="https://github.com/user-attachments/assets/805bd804-6d45-484e-9d6c-2bad635ae82f" />

### 3. Disease Detection  
- Image upload  
- EfficientNet inference  
- Confidence scores  
- Recommendation engine  

<img width="1035" height="480" alt="detection" src="https://github.com/user-attachments/assets/da984ec2-c10f-4d67-ad19-6762588f10a5" />


### 4. Weather Forecasting  
- OpenWeatherMap API  
- 7-day forecast  
- Alerts for humidity, rainfall, wind  

<img width="1353" height="632" alt="Wheather" src="https://github.com/user-attachments/assets/dfa0a7ba-5e25-483d-935a-cfe14124fdd0" />

### 5 Days weather Forcasting 
<img width="1350" height="635" alt="Wheather 2" src="https://github.com/user-attachments/assets/b7652d3c-67a4-4636-a257-cfcd05537c41" />

## 5. Multilingual Chatbot  
- Supports English, Hindi, Marathi, Tamil, Telugu  
- Agricultural Q&A  
- Context-aware responses  
<img width="800" height="600" alt="ChatBot" src="https://github.com/user-attachments/assets/04cb1a92-28b6-44d3-9af0-e6bed234a77a" />

### Multilingual chatbot for Farmers in India for Different Differnt Regional Langugae
![WhatsApp Image 2025-11-27 at 11 39 49_87471b55](https://github.com/user-attachments/assets/4180a62b-8746-4def-b607-2a64b055c6a6)



## Tools & Technologies

| Category | Technologies |
|---------|--------------|
| Backend | Python Flask |
| Deep Learning | TensorFlow, Keras |
| Model | EfficientNet-B3 |
| Frontend | HTML, CSS, JS, Bootstrap |
| Database | SQLite / PostgreSQL |
| APIs | OpenWeatherMap, NLP Chatbot |
| Version Control | Git & GitHub |



## Results & Impact

### Quantitative
- **94.2% validation accuracy**  
- **180ms inference time**  
- Tested with **150+ real-world queries**  

### Qualitative
- Accessible to farmers with low digital literacy  
- Reduces crop loss via early disease identification  
- Multilingual support increases usability  



## Future Scope
- Support for 20+ crop varieties  
- Mobile app (Android/iOS)  
- IoT integration (soil sensors)  
- Drone image analysis  
- Yield prediction and geospatial disease mapping  



## Acknowledgements

Special thanks to **Dr. Jinesh Melvin Y I**, Project Guide at *Pillai College of Engineering, Panvel*, for his guidance and support.

Thanks to project team members:  
- **Ashwin Baburaj**  
- **Yash Karande**  
- **Vuvarj Kolekar**  

#### *Form*
Project Leader : Adhish Biju
---

## Conclusion

AgroAI Assist demonstrates how AI can transform agriculture by enabling early disease detection, weather-aware farming, and multilingual support. The system’s publication in the **Vesper Journal (Scopus Indexed)** highlights its research depth and practical significance in precision agriculture.

<div align="center">

**Project Status:** Completed & Published  
**Publication:** Vesper Journal 
**Name** Adhish Biju
**Contact:** [GmaiL: adhishbiju2000@gmail.com]  

</div>


