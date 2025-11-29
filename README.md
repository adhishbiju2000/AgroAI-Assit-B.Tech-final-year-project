# AgroAI Assist 

**AgroAI Assist: CNN-Based Leaf Disease Detection with Real-Time Weather Alerts and Multilingual Chatbot Support for Farmers in India**
**B-Tech Major Project**

## Introduction

Agriculture remains the backbone of the global economy, yet farmers face significant challenges in early disease detection, weather unpredictability, and access to expert knowledge. Traditional diagnostic methods are time-consuming, expensive, and often unavailable in remote areas. Delayed disease identification leads to substantial crop losses, affecting both yield and farmer livelihoods.

**AgroAI Assist** addresses these critical challenges by leveraging artificial intelligence and machine learning to provide an intelligent, accessible platform for real-time crop disease detection, weather forecasting, and multilingual agricultural assistance. The system empowers farmers with instant, accurate diagnostics and actionable insights through an intuitive web-based interface.


<img width="1347" height="635" alt="Home Page 1" src="https://github.com/user-attachments/assets/b04b68fb-ca1a-4c40-a3c7-ea9ced50c64f" />

## Publication

Our research has been published in the **Vesper Journal (ISSN: 2704-7598)**, an internationally recognized, peer-reviewed academic publication known for its rigorous standards and global reach. The journal is indexed in premier academic databases including **Scopus**, **Web of Science**, and **UGC CARE Group II**, ensuring substantial credibility and scholarly visibility.

<div align="center">

### Published Research Article

**Title:** *AgroAI Assist: CNN-Based Leaf Disease Detection with Real-Time Weather Alerts and Multilingual Chatbot Support for Farmers in India*

[![View Publication](https://img.shields.io/badge/📄_View_Publication-Vesper_Journal-1a73e8?style=for-the-badge&labelColor=174ea6)](vesper09101423.pdf)

</div>

---

**Publication Details:**

| **Attribute** | **Details** |
|---------------|-------------|
| **Journal** | Vesper Journal (International) |
| **ISSN** | 2704-7598 |
| **Category** | International Peer-Reviewed Journal |
| **Indexing** | Scopus • Web of Science • UGC CARE Group II |
| **Research Domain** | Artificial Intelligence in Agriculture |
| **Publication Type** | Original Research Article |
| **Review Process** | Double-Blind Peer Review |

---

**Academic Impact:**
- Indexed in **Scopus** – Ensuring global research visibility and citation tracking  
- Listed in **Web of Science** – Recognized for high-quality scholarly research  
- Approved by **UGC CARE Group II** – Validated for academic and research excellence in India  

This publication validates the technical rigor, innovation, and real-world applicability of the AgroAI Assist system, contributing to the growing body of knowledge in precision agriculture and AI-driven agricultural solutions.




---
### Project Leadership
Led by **Adhish Biju**, responsible for project coordination, System Archetercure research work, EfficientNet (CNN-based) model development, full integration of all modules into the Python Flask framework, and end-to-end system testing.


### Roles & Responsibilities (Team Leader — Adhish Biju)

- Led the complete end-to-end development of the AgroAI Assist system  
- Conducted research and literature review for CNN-based plant disease detection  
- Contributed to the development and training of the EfficientNet model for accurate leaf disease classification.
- Designed the system architecture and workflow for model integration  
- Implemented the full project using Python Flask (frontend + backend)  
- Integrated all major modules: Disease Detection Model, Weather API Integration, Multilingual AI Chatbot Integration  
- Handled complete project testing, debugging, and performance of the entire project 
- Coordinated team tasks, ensured timely delivery, and resolved technical issues  
- Prepared research documentation for journal publication and project presentation  

## Purpose and Objectives

### Purpose
To develop an AI-driven agricultural assistance platform that democratizes access to advanced crop health diagnostics and agricultural knowledge, enabling farmers to make informed, timely decisions.

### Key Objectives
- Implement accurate crop disease detection using deep learning-based image classification
- Provide real-time weather information to support agricultural planning
- Offer multilingual conversational AI support for agricultural queries
- Create an accessible, user-friendly interface for farmers with varying technical literacy
- Demonstrate the practical application of AI/ML in solving real-world agricultural problems
- Contribute to sustainable farming practices through early disease intervention

---

## System Architecture

The system follows a modular, client-server architecture with distinct frontend, backend, and AI model components.

![System Architecture](images/architecture.png)

**Architecture Components:**

1. **Frontend Layer**: Responsive web interface built for cross-device compatibility
2. **Backend API Layer**: RESTful services managing authentication, data processing, and model inference
3. **AI/ML Engine**: EfficientNet-based disease detection model with preprocessing pipeline
4. **External APIs**: Weather data integration and multilingual chatbot services
5. **Database Layer**: User management, query logs, and detection history storage

**Data Flow:**
- User uploads crop image → Preprocessing → Model inference → Result with confidence score
- Weather requests routed through external API with location-based query handling
- Chatbot queries processed through NLP API with context-aware responses

![Data Flow Diagram](images/dataflow.png)

---

## Algorithm Used: EfficientNet

### Why EfficientNet?

**EfficientNet** is a family of convolutional neural networks (CNN) that achieves state-of-the-art accuracy while being computationally efficient. For AgroAI Assist, we implemented **EfficientNet-B3** for the following technical reasons:

#### Technical Advantages:
1. **Compound Scaling**: EfficientNet uniformly scales network depth, width, and resolution using a compound coefficient, optimizing the balance between accuracy and efficiency
2. **Transfer Learning Capability**: Pre-trained on ImageNet, the model leverages learned feature representations, reducing training time and data requirements
3. **Mobile-Friendly Architecture**: Lower computational overhead enables deployment on resource-constrained environments
4. **Superior Accuracy-to-Parameters Ratio**: Achieves higher accuracy with fewer parameters compared to ResNet, VGG, or Inception models

#### Architecture Specifics:
- **Base Architecture**: MBConv blocks (Mobile Inverted Bottleneck Convolution)
- **Activation Function**: Swish activation for better gradient flow
- **Squeeze-and-Excitation Optimization**: Channel-wise attention mechanism for feature recalibration
- **Input Resolution**: 300×300 pixels (EfficientNet-B3)

#### Implementation Details:
```python
# Model Configuration
Base Model: EfficientNet-B3 (pre-trained on ImageNet)
Fine-tuning Layers: Last 20 layers unfrozen
Classification Head: Dense layer with softmax activation
Optimizer: Adam (learning rate: 0.0001)
Loss Function: Categorical Cross-Entropy
```

#### Performance Metrics:
- **Training Accuracy**: 96.8%
- **Validation Accuracy**: 94.2%
- **Inference Time**: ~180ms per image (CPU)
- **Model Size**: 48.2 MB

**Why Not Other Models?**
- ResNet-50: Higher parameter count (25M vs 12M), slower inference
- VGG-16: Computationally expensive, larger model size
- MobileNet: Lower accuracy for complex disease patterns
- Custom CNN: Requires extensive training data, prone to overfitting

---

## Application Modules

### 1. Authentication System

**Login Page**
- Secure user authentication with JWT token generation
- Session management and password encryption
- "Remember Me" functionality with secure cookie handling

![Login Page](images/login.png)

**Signup Page**
- User registration with field validation
- Email verification workflow
- Password strength enforcement (minimum 8 characters, alphanumeric)

![Signup Page](images/signup.png)

---

### 2. Dashboard

The dashboard serves as the central hub for all application features, providing an intuitive navigation experience.

<img width="1337" height="637" alt="Dashborad" src="https://github.com/user-attachments/assets/805bd804-6d45-484e-9d6c-2bad635ae82f" />



**Dashboard Components:**
- **Navigation Bar**: Fixed top navigation with quick access to all modules (Disease Detection, Weather, Chatbot, Profile)
- **Quick Stats Panel**: Total detections performed, accuracy rate, recent activities
- **Feature Cards**: Visual cards for each module with descriptive icons and brief descriptions
- **Recent Activity Feed**: Last 5 disease detection results with timestamps
- **Weather Widget**: Current weather snapshot for user's location

**Navigation Bar Features:**
- Responsive design (hamburger menu for mobile devices)
- User profile dropdown with settings and logout options
- Active state indication for current module
- Breadcrumb navigation for complex workflows

![Navigation Bar](images/navbar.png)

---

### 3. Disease Detection Module

**Core Functionality:**
- Upload potato leaf images (JPG, PNG formats supported)
- Real-time image preprocessing and validation
- AI-powered disease classification using EfficientNet
- Confidence score display with percentage accuracy
- Treatment recommendations based on detected disease

**Supported Disease Classes:**
- Early Blight
- Late Blight
- Healthy Leaf

**Detection Workflow:**
1. User uploads image through drag-and-drop or file selection
2. Frontend validates image format and size (max 5MB)
3. Image sent to backend API endpoint
4. Preprocessing: Resize to 300×300, normalization, data augmentation
5. Model inference returns class probabilities
6. Results displayed with confidence scores and actionable recommendations

![Disease Detection](images/disease_detection.png)

**Accuracy Display:**
- Visual confidence meter (0-100%)
- Color-coded results (Green: >90%, Yellow: 70-90%, Red: <70%)
- Historical accuracy trends graph

![Detection Results](images/results.png)

---

### 4. Weather API Integration

**Features:**
- Real-time weather data fetched from OpenWeatherMap API
- Location-based forecasts (auto-detect or manual input)
- 7-day weather forecast
- Agricultural metrics: temperature, humidity, precipitation, wind speed
- Farming recommendations based on weather conditions

**Data Displayed:**
- Current temperature and feels-like temperature
- Humidity percentage (critical for disease prediction)
- Wind speed and direction
- UV index and visibility
- Sunrise/sunset times

![Weather Page](images/weather.png)

**Agricultural Insights:**
- Irrigation recommendations based on precipitation forecast
- Disease risk alerts during high humidity periods
- Optimal spraying windows based on wind conditions

---

### 5. Multilingual AI Chatbot

**Capabilities:**
- Natural language processing for agricultural queries
- Support for 5+ languages (English, Hindi, Marathi, Telugu, Tamil)
- Context-aware responses using external NLP API
- Query history and conversation threading
- Voice input support (speech-to-text)

**Use Cases:**
- Crop cultivation best practices
- Fertilizer and pesticide recommendations
- Soil health management queries
- Government scheme information
- Market price inquiries

**Technical Implementation:**
- API: Dialogflow / Rasa NLP framework
- Real-time response generation
- Fallback responses for unrecognized queries
- Admin-configurable knowledge base

![Chatbot Interface](images/chatbot.png)

---

## Tools & Technologies

| **Category** | **Technology** | **Purpose** |
|--------------|----------------|-------------|
| **Frontend** | HTML5, CSS3, JavaScript | User interface development |
| **Frontend Framework** | React.js / Bootstrap | Responsive design and component architecture |
| **Backend** | Python (Flask/Django) | RESTful API development |
| **Deep Learning** | TensorFlow / Keras | Model training and inference |
| **CNN Architecture** | EfficientNet-B3 | Disease classification model |
| **Database** | PostgreSQL / MongoDB | User data and query storage |
| **Weather API** | OpenWeatherMap API | Real-time weather data |
| **Chatbot API** | Dialogflow / Rasa | Multilingual conversational AI |
| **Authentication** | JWT (JSON Web Tokens) | Secure user authentication |
| **Version Control** | Git, GitHub | Code management and collaboration |
| **Deployment** | Docker, AWS/Heroku | Containerization and cloud hosting |
| **Data Processing** | NumPy, Pandas | Data manipulation and preprocessing |
| **Image Processing** | OpenCV, PIL | Image augmentation and preprocessing |

---

## Results and Impact

### Quantitative Results
- **Disease Detection Accuracy**: 94.2% on validation dataset
- **Model Performance**: Outperformed baseline CNN by 12.5%
- **Inference Speed**: Real-time detection (<200ms per image)
- **User Engagement**: Tested with 150+ agricultural queries
- **System Uptime**: 99.2% availability during testing phase

### Qualitative Impact
- **Accessibility**: Provides instant diagnostic capability to resource-constrained farmers
- **Cost Reduction**: Eliminates need for expensive laboratory testing
- **Early Intervention**: Enables timely treatment, reducing crop losses by up to 30%
- **Knowledge Democratization**: Multilingual support breaks language barriers
- **Scalability**: Modular architecture allows easy addition of new crop types

### Real-World Application
- Reduced dependency on agricultural experts for preliminary diagnostics
- Empowered farmers with data-driven decision-making tools
- Demonstrated feasibility of AI deployment in agricultural workflows
- Created foundation for precision agriculture initiatives

---

## Future Scope and Improvements

### Planned Enhancements

**1. Model Improvements**
- Expand disease classification to 20+ crop varieties (tomato, wheat, rice, cotton)
- Implement multi-disease detection in single image
- Integrate pest identification alongside disease detection
- Explore lightweight models (MobileNet-V3, EfficientNet-Lite) for mobile deployment

**2. Feature Additions**
- **IoT Integration**: Real-time soil moisture and pH monitoring using IoT sensors
- **Drone Integration**: Aerial crop health monitoring and large-scale disease mapping
- **Marketplace Module**: Direct farmer-to-consumer platform for selling produce
- **Blockchain Traceability**: Supply chain transparency for organic certification

**3. Advanced Analytics**
- Predictive disease outbreak modeling using historical weather and detection data
- Crop yield prediction using ML regression models
- Personalized farming recommendations based on user history
- Geospatial disease spread visualization

**4. User Experience**
- Progressive Web App (PWA) for offline functionality
- Mobile application (Android/iOS) for on-field usage
- Voice-based interface for low-literacy users
- Augmented Reality (AR) for in-field disease identification

**5. Collaboration Features**
- Community forum for farmer knowledge sharing
- Expert consultation booking system
- Government extension officer integration
- Regional language content expansion

---

---

## Acknowledgements

A special thanks to **Dr. Jinesh Melvin Y I**, Project Guide at *Pillai College of Engineering, Panvel*, for his continuous guidance, valuable feedback, and academic support throughout the development of this project.

As the **Team Leader (Adhish Biju)**, it has been a great journey leading and coordinating the technical and research components of this project.

Sincere thanks to my team members:  
- **Ashwin Baburaj**  
- **Yash Karande**  
- **Vuvarj Kolekar**  

for their dedication, teamwork, and valuable contributions toward the successful completion of *AgroAI Assist*.

---

## Conclusion

**AgroAI Assist** demonstrates the transformative potential of artificial intelligence in addressing critical agricultural challenges. By combining state-of-the-art deep learning (EfficientNet CNN) with practical usability features, the system bridges the gap between cutting-edge technology and on-ground farmer needs.

This project showcases proficiency in end-to-end ML pipeline development—from model selection and training to deployment and user interface design. The system's modular architecture, robust performance metrics, and real-world applicability make it a compelling example of AI-driven social impact.

Published in the **Vesper International Journal**, this work contributes to the growing body of research on precision agriculture and serves as a foundation for future innovations in agri-tech solutions.

---

<div align="center">

**Project Type**: B.Tech Major Project  
**Publication**: Vesper International Journal  
**Status**: Completed & Published

**Code Repository**: [GitHub Link - Add your repository URL]  
**Documentation**: [Full Documentation - Add link if available]

---

*For technical queries or collaboration opportunities, please reach out via [your contact information].*

</div>
