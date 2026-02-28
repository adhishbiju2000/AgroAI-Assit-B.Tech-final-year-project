import os
import sqlite3
import random
import string
import re
import torch
import torch.nn.functional as F
import timm
import cv2
import numpy as np
import io
import base64
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from gtts import gTTS
from groq import Groq
import markdown
from torchvision import transforms
from PIL import Image

# ----------------- APP CONFIG -----------------
app = Flask(__name__)
app.secret_key = "supersecretkey"  # Change in production
app.config['UPLOAD_FOLDER'] = 'Uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static/audio', exist_ok=True)
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'm4a'}

# Initialize Groq client
groq_client = Groq(api_key="")

# ----------------- MODEL SETUP -----------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
POTATO_MODEL_PATH = "best_model_potato_updated.pth"
TOMATO_MODEL_PATH = "best_tomato_model.pth"  # Assuming this is the path to the tomato model

potato_model = None
tomato_model = None

try:
    potato_model = timm.create_model("tf_efficientnet_b0", pretrained=False, num_classes=7)
    state_dict = torch.load(POTATO_MODEL_PATH, map_location=DEVICE)
    potato_model.load_state_dict(state_dict)
    potato_model = potato_model.to(DEVICE)
    potato_model.eval()
    print("Potato disease detection model loaded successfully!")
except Exception as e:
    print(f"Warning: Could not load potato disease detection model: {e}")
    print("Potato disease detection features will be disabled.")

try:
    tomato_model = timm.create_model("tf_efficientnet_b0", pretrained=False, num_classes=10)
    state_dict = torch.load(TOMATO_MODEL_PATH, map_location=DEVICE)
    tomato_model.load_state_dict(state_dict)
    tomato_model = tomato_model.to(DEVICE)
    tomato_model.eval()
    print("Tomato disease detection model loaded successfully!")
except Exception as e:
    print(f"Warning: Could not load tomato disease detection model: {e}")
    print("Tomato disease detection features will be disabled.")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

potato_class_names = [
    "Early Blight",
    "Fungal Diseases", 
    "Healthy",
    "Late Blight",
    "Plant Pests",
    "Potato Cyst Nematode",
    "Potato Virus"
]

tomato_class_names = [
    'Tomato_Bacterial_spot', 
    'Tomato_Early_blight', 
    'Tomato_Late_blight', 
    'Tomato_Leaf_Mold', 
    'Tomato_Septoria_leaf_spot', 
    'Tomato_Spider_mites Two-spotted_spider_mite', 
    'Tomato_Target_Spot', 
    'Tomato_Tomato_Yellow_Leaf_Curl_Virus', 
    'Tomato_Tomato_mosaic_virus', 
    'Tomato_healthy'
]

CLASS_COLORMAPS = {
    "Early Blight": cv2.COLORMAP_JET,
    "Fungal Diseases": cv2.COLORMAP_OCEAN,
    "Healthy": None,
    "Late Blight": cv2.COLORMAP_TURBO,
    "Plant Pests": cv2.COLORMAP_HOT,
    "Potato Cyst Nematode": cv2.COLORMAP_SPRING,
    "Potato Virus": cv2.COLORMAP_WINTER,
    'Tomato_Bacterial_spot': cv2.COLORMAP_BONE,
    'Tomato_Early_blight': cv2.COLORMAP_OCEAN,
    'Tomato_Late_blight': cv2.COLORMAP_WINTER,
    'Tomato_Leaf_Mold': cv2.COLORMAP_HOT,
    'Tomato_Septoria_leaf_spot': cv2.COLORMAP_TWILIGHT,
    'Tomato_Spider_mites Two-spotted_spider_mite': cv2.COLORMAP_INFERNO,
    'Tomato_Target_Spot': cv2.COLORMAP_PLASMA,
    'Tomato_Tomato_Yellow_Leaf_Curl_Virus': cv2.COLORMAP_VIRIDIS,
    'Tomato_Tomato_mosaic_virus': cv2.COLORMAP_MAGMA,
    'Tomato_healthy': None,
}

# ----------------- DISEASE INFORMATION DATABASE -----------------
DISEASE_INFO = {
    "Early Blight": {
        "descriptions": [
            "Early blight is a destructive fungal disease caused by Alternaria solani that primarily affects potato and tomato plants. This pathogen thrives in warm, humid conditions and can significantly reduce crop yields.",
            "This fungal infection, scientifically known as Alternaria solani, manifests as distinctive dark brown spots with concentric rings on plant leaves. It's one of the most common diseases affecting solanaceous crops worldwide.",
            "Early blight disease is characterized by its target-like lesions that appear on older leaves first. The fungal pathogen Alternaria solani spreads rapidly during periods of high humidity and moderate temperatures.",
            "Caused by the fungus Alternaria solani, early blight creates characteristic bull's-eye patterns on leaves. This disease can cause severe defoliation and reduce photosynthetic capacity of the plant.",
            "Early blight is a necrotrophic fungal disease that affects the foliage, stems, and fruits of potato plants. The pathogen Alternaria solani produces toxins that kill plant tissue, creating the distinctive lesions."
        ],
        "reasons": [
            "High humidity levels (above 90%) combined with temperatures between 24-29°C create ideal conditions for spore germination and infection. Poor air circulation and dense plant canopies exacerbate the problem.",
            "Prolonged leaf wetness from dew, rain, or overhead irrigation provides the moisture needed for fungal spore development. Stressed plants due to nutrient deficiencies are more susceptible to infection.",
            "The fungus overwinters in infected plant debris and soil, releasing spores during favorable weather conditions. Wind and rain splash spread the spores to healthy plants, initiating new infection cycles.",
            "Excessive nitrogen fertilization can make plants more susceptible by promoting lush growth with tender tissues. Potassium deficiency also weakens plant resistance to fungal pathogens.",
            "Poor crop rotation practices allow the pathogen to build up in soil. Growing susceptible crops in the same field repeatedly increases disease pressure and reduces natural soil suppressiveness."
        ],
        "precautions": [
            "Implement proper crop rotation with non-solanaceous crops for at least 2-3 years. Remove and destroy infected plant debris immediately after harvest to reduce inoculum sources.",
            "Ensure adequate plant spacing (45-60cm apart) to improve air circulation and reduce humidity around plants. Avoid overhead irrigation and water plants at soil level during early morning hours.",
            "Apply preventive fungicide sprays containing copper compounds or chlorothalonil before disease symptoms appear. Begin applications when environmental conditions favor disease development.",
            "Use certified disease-free seeds and resistant varieties when available. Maintain balanced nutrition with adequate potassium and avoid excessive nitrogen fertilization that promotes susceptible growth.",
            "Install drip irrigation systems to minimize leaf wetness duration. Stake or cage plants properly to improve air circulation and prevent soil splash onto lower leaves during rainfall."
        ],
        "pesticides": [
            "Systemic fungicides: Propiconazole (Tilt) at 0.1% concentration, apply every 10-14 days. Azoxystrobin (Quadris) at 250-500ml per hectare provides excellent protection against early blight.",
            "Contact fungicides: Mancozeb (Dithane M-45) at 2-2.5g per liter of water. Copper oxychloride (Blitox) at 3g per liter provides good preventive control when applied regularly.",
            "Combination products: Ridomil Gold MZ (metalaxyl + mancozeb) at 2.5g per liter offers both preventive and curative action. Apply at 7-10 day intervals during favorable conditions.",
            "Biological control: Bacillus subtilis-based products like Serenade at 2-4ml per liter. Trichoderma harzianum formulations can suppress fungal growth when applied as soil treatment.",
            "Organic options: Neem oil at 3-5ml per liter combined with potassium bicarbonate at 1g per liter. Bordeaux mixture (copper sulfate + lime) at 1% concentration for organic farming systems."
        ]
    },
    "Fungal Diseases": {
        "descriptions": [
            "Fungal diseases represent a diverse group of plant pathogens that can cause various symptoms including leaf spots, blights, rots, and wilts. These organisms reproduce through spores and thrive in moist conditions.",
            "This category encompasses multiple fungal pathogens that attack different parts of the plant. Common genera include Fusarium, Rhizoctonia, Pythium, and Phytophthora, each causing distinct symptoms and damage patterns.",
            "Fungal infections in crops can manifest as powdery mildews, downy mildews, rusts, or various leaf spot diseases. These pathogens often have complex life cycles involving different spore stages.",
            "Various fungal species can simultaneously or sequentially infect plants, creating complex disease scenarios. The interaction between different fungi can sometimes intensify disease severity and complicate management strategies.",
            "Fungal diseases are among the most economically important plant health issues globally. They can affect seed germination, root development, nutrient uptake, photosynthesis, and overall plant vigor."
        ],
        "reasons": [
            "Warm, humid weather conditions (relative humidity >80%) favor most fungal pathogens. Temperature fluctuations combined with moisture create stress conditions that weaken plant immunity.",
            "Poor soil drainage and waterlogged conditions promote root rot fungi like Pythium and Phytophthora. Compacted soils with limited oxygen availability favor anaerobic pathogenic fungi.",
            "Contaminated seeds, tools, or planting materials introduce fungal pathogens to healthy fields. Infected plant debris left in fields serves as a primary inoculum source for subsequent crops.",
            "Imbalanced nutrition, particularly nitrogen excess and potassium deficiency, makes plants more susceptible to fungal attacks. Stressed plants due to drought or other factors have compromised defense mechanisms.",
            "Dense plant populations with poor air circulation create microclimates favorable for fungal development. Overhead irrigation systems that keep foliage wet for extended periods promote spore germination."
        ],
        "precautions": [
            "Maintain proper field hygiene by removing crop residues and weeds that harbor fungal pathogens. Practice clean cultivation with sanitized tools and equipment to prevent pathogen spread.",
            "Improve soil drainage through proper land preparation and organic matter addition. Avoid overwatering and ensure fields have adequate drainage systems to prevent waterlogging.",
            "Use certified pathogen-free seeds and planting materials. Treat seeds with fungicidal dressings before planting to eliminate seed-borne pathogens and provide early protection.",
            "Implement integrated disease management combining resistant varieties, proper nutrition, and timely fungicide applications. Monitor weather conditions and apply preventive treatments before conducive periods.",
            "Establish proper plant spacing and row orientation to maximize air circulation and sunlight penetration. Use mulching to prevent soil splash and reduce humidity around plant bases."
        ],
        "pesticides": [
            "Broad-spectrum fungicides: Tebuconazole (Folicur) at 1ml per liter provides excellent control against multiple fungal pathogens. Propiconazole (Tilt) at 0.1% offers systemic protection.",
            "Preventive treatments: Chlorothalonil (Kavach) at 2ml per liter applied at 10-day intervals. Mancozeb (Dithane M-45) at 2.5g per liter for contact protection against spores.",
            "Systemic fungicides: Hexaconazole (Contaf) at 2ml per liter for internal plant protection. Carbendazim (Bavistin) at 1g per liter for systemic movement and long-lasting control.",
            "Combination products: Cymoxanil + Mancozeb (Curzate M) at 2g per liter for both contact and systemic action. Metalaxyl + Mancozeb (Ridomil Gold) for soil and foliar application.",
            "Resistance management: Rotate between different chemical groups - triazoles, strobilurins, and multi-site fungicides. Use tank mixtures to delay resistance development in pathogen populations."
        ]
    },
    "Healthy": {
        "descriptions": [
            "Excellent news! Your plant appears to be in optimal health with no visible signs of disease or pest damage. The foliage shows vibrant green coloration and proper leaf structure typical of healthy plant tissue.",
            "Congratulations! The analysis indicates your crop is thriving with no detectable diseases. Healthy plants exhibit strong cellular structure, proper chlorophyll content, and absence of pathological symptoms.",
            "Your plant sample shows all indicators of robust health - proper leaf color, intact cell structure, and no evidence of pathogenic organisms. This suggests effective management practices are in place.",
            "The diagnostic results confirm your plant is disease-free and maintaining excellent physiological condition. Healthy plants like this demonstrate good nutrition, adequate water management, and stress-free growing conditions.",
            "Perfect! Your crop shows optimal health parameters with no signs of fungal, bacterial, or viral infections. The plant tissue appears strong and resilient, indicating successful preventive management strategies."
        ],
        "reasons": [
            "Proper cultural practices including balanced nutrition, appropriate irrigation, and good field hygiene have maintained plant health. Regular monitoring and preventive measures have prevented disease establishment.",
            "Favorable environmental conditions with moderate temperatures, adequate moisture, and good air circulation support plant health. The absence of stress factors has allowed natural disease resistance to function effectively.",
            "Use of disease-resistant varieties combined with integrated pest management practices has protected the crop from major diseases. Proper crop rotation has reduced pathogen pressure in the field.",
            "Timely preventive treatments and maintenance of optimal growing conditions have kept the plant healthy. Good soil health with beneficial microorganisms supports natural disease suppression.",
            "Effective farm management including clean cultivation, proper spacing, and regular field inspection has prevented disease development. Balanced ecosystem with natural enemies controls potential pests."
        ],
        "precautions": [
            "Continue current management practices to maintain plant health. Regular field monitoring for early disease detection remains crucial even in healthy crops.",
            "Maintain preventive fungicide application schedule during disease-favorable weather conditions. Don't become complacent - early intervention prevents major disease outbreaks.",
            "Keep practicing good agricultural hygiene including tool sanitization, debris removal, and weed control. These practices prevent introduction and buildup of pathogenic organisms.",
            "Continue balanced fertilization program avoiding nitrogen excess that can predispose plants to diseases. Maintain adequate potassium and micronutrient levels for strong plant immunity.",
            "Monitor weather conditions and be prepared to implement additional protective measures during high-risk periods. Maintain spray equipment in ready condition for emergency applications."
        ],
        "pesticides": [
            "Preventive maintenance: Continue light protective sprays with copper compounds at 15-day intervals during humid periods. This maintains a protective barrier against potential infections.",
            "Biological protection: Apply beneficial microorganism preparations like Trichoderma or Bacillus subtilis monthly to enhance soil and plant health. These promote natural disease suppression.",
            "Nutrient sprays: Use foliar applications of potassium phosphonate at 2ml per liter to boost plant immunity. Calcium chloride sprays (2g per liter) strengthen cell walls against pathogen entry.",
            "Organic maintenance: Neem oil at 3ml per liter every 2 weeks provides broad-spectrum protection while being environmentally friendly. Seaweed extracts enhance plant natural defense mechanisms.",
            "Systemic protection: Light applications of systemic fungicides like propiconazole (0.05%) during critical growth stages provide insurance against sudden disease outbreaks without being excessive."
        ]
    },
    "Late Blight": {
        "descriptions": [
            "Late blight is a devastating oomycete disease caused by Phytophthora infestans, historically responsible for the Irish Potato Famine. This water mold pathogen can destroy entire crops within days under favorable conditions.",
            "This aggressive plant pathogen, Phytophthora infestans, causes water-soaked lesions that rapidly expand on leaves and stems. The disease is particularly destructive during cool, moist weather conditions typical of highland areas.",
            "Late blight manifests as dark, irregular patches on leaves with white fuzzy growth on the undersides. The pathogen Phytophthora infestans is not a true fungus but an oomycete with unique characteristics and treatment requirements.",
            "Phytophthora infestans, the causal agent of late blight, produces sporangia that can spread rapidly through wind and rain. This disease can affect all parts of the potato plant including tubers, causing storage rot.",
            "Late blight disease progresses rapidly from initial infection to complete plant death within 7-10 days. The pathogen Phytophthora infestans thrives in cool temperatures (15-25°C) with high humidity."
        ],
        "reasons": [
            "Cool temperatures (15-25°C) combined with high humidity (>90%) and frequent rainfall create perfect conditions for late blight development. Extended leaf wetness periods allow sporangial germination.",
            "The pathogen survives in infected seed tubers and volunteer potatoes, providing primary inoculum for new infections. Wind-dispersed sporangia can travel several kilometers to infect new fields.",
            "Dense plant canopies with poor air circulation maintain high humidity microclimates favorable for disease development. Overhead irrigation systems prolong leaf wetness and facilitate spore movement.",
            "Late planting during monsoon seasons increases disease risk due to conducive weather conditions. Fields located in valleys or areas with poor drainage are particularly susceptible.",
            "Use of susceptible varieties without resistance genes makes crops vulnerable to infection. Inadequate crop rotation allows pathogen buildup in soil and crop debris."
        ],
        "precautions": [
            "Plant certified disease-free seed tubers from reputable sources. Avoid saving tubers from infected fields as they can carry the pathogen to the next season.",
            "Implement strict field sanitation by destroying infected plant debris immediately. Remove volunteer potato plants that can serve as green bridges for the pathogen.",
            "Apply preventive fungicide sprays before disease symptoms appear, especially during favorable weather conditions. Use weather-based disease forecasting systems to time applications.",
            "Improve field drainage and avoid overhead irrigation. Plant in well-ventilated areas with good air circulation to reduce humidity around plants.",
            "Choose resistant varieties when available and practice crop rotation with non-host crops for at least 2 years. Monitor fields regularly during high-risk periods."
        ],
        "pesticides": [
            "Systemic fungicides: Metalaxyl-M + Mancozeb (Ridomil Gold MZ) at 2.5g per liter provides excellent control. Apply at 7-day intervals during disease-favorable conditions.",
            "Contact fungicides: Propineb (Antracol) at 2.5g per liter or Metiram (Polyram) at 2g per liter. These provide protective barriers against spore germination on leaf surfaces.",
            "Newer chemistries: Cyazofamid (Ranman) at 1ml per liter offers specific activity against oomycetes. Mandipropamid (Revus) at 1.25ml per liter provides excellent late blight control.",
            "Combination treatments: Cymoxanil + Famoxadone (Equation Pro) at 2ml per liter combines contact and systemic activity. Dimethomorph + Ametoctradin (Lenko) for resistance management.",
            "Emergency control: Copper compounds like copper hydroxide at 3g per liter can slow disease progress. Fosetyl-Al (Aliette) at 2.5g per liter stimulates plant defense mechanisms."
        ]
    },
    "Plant Pests": {
        "descriptions": [
            "Plant pests encompass a wide range of arthropods including aphids, thrips, whiteflies, and mites that feed on plant tissues. These insects can cause direct damage through feeding and indirect damage by transmitting plant viruses.",
            "Various insect pests attack different parts of the plant, from root-feeding grubs to leaf-chewing caterpillars and sap-sucking bugs. Each pest type requires specific identification and targeted management strategies.",
            "Pest infestations can rapidly build up under favorable conditions, causing significant economic losses through reduced yield and quality. Some pests also serve as vectors for bacterial and viral diseases.",
            "Insect pests display diverse feeding behaviors including piercing-sucking, chewing, mining, and boring. Understanding pest biology and behavior is crucial for developing effective integrated pest management programs.",
            "Multi-species pest complexes often occur simultaneously, requiring comprehensive management approaches. Beneficial insects and natural enemies play important roles in pest population regulation in healthy ecosystems."
        ],
        "reasons": [
            "Warm weather conditions accelerate pest reproduction rates and development cycles. Many insects complete multiple generations per season under favorable temperature and humidity conditions.",
            "Monoculture cropping systems provide abundant food sources for specialist pests, allowing populations to build up rapidly. Lack of crop diversity reduces natural enemy populations.",
            "Excessive nitrogen fertilization produces succulent plant tissues that are more attractive to many pest species. Stressed plants due to drought or other factors may also be more susceptible.",
            "Destruction of natural habitats and overuse of broad-spectrum insecticides eliminates beneficial insects that naturally control pest populations. This leads to secondary pest outbreaks.",
            "Introduction of exotic pests without natural enemies can cause severe infestations. Climate change and global trade facilitate pest movement and establishment in new regions."
        ],
        "precautions": [
            "Implement regular field scouting to monitor pest populations and identify species correctly. Early detection allows for timely intervention before populations reach economic thresholds.",
            "Practice crop rotation and intercropping to break pest life cycles and support natural enemy populations. Maintain field borders with beneficial flowering plants that provide habitat for predators and parasites.",
            "Use pheromone traps and yellow sticky traps for monitoring and mass trapping of specific pests. These tools help determine optimal timing for control measures.",
            "Maintain balanced plant nutrition avoiding excessive nitrogen that promotes pest-attractive growth. Healthy, well-nourished plants often have better natural resistance to pest attacks.",
            "Preserve beneficial insects by using selective pesticides when necessary and avoiding applications during pollinator active periods. Create refuge areas with diverse flowering plants."
        ],
        "pesticides": [
            "Selective insecticides: Imidacloprid (Confidor) at 0.5ml per liter for aphids and whiteflies. Spinosad (Success) at 1ml per liter targets lepidopteran larvae while preserving beneficials.",
            "Contact insecticides: Lambda-cyhalothrin (Karate) at 1ml per liter provides quick knockdown of various pests. Bifenthrin (Talstar) at 1ml per liter offers residual control.",
            "Systemic treatments: Thiamethoxam (Actara) at 0.3g per liter for sucking pests. Acetamiprid (Pridee) at 0.5ml per liter provides good control of resistant aphid populations.",
            "Biological controls: Bacillus thuringiensis (Bt) formulations for caterpillar control at 1-2ml per liter. Beauveria bassiana fungal preparations for various insect pests.",
            "Natural products: Neem-based insecticides at 3-5ml per liter provide broad-spectrum control while being relatively safe to beneficials. Pyrethrin formulations for immediate knockdown effect."
        ]
    },
    "Potato Cyst Nematode": {
        "descriptions": [
            "Potato cyst nematodes (Globodera species) are microscopic roundworms that form protective cysts on potato roots. These soil-borne parasites can survive in soil for decades without a host plant.",
            "These specialized plant-parasitic nematodes penetrate potato roots and develop into sedentary females that swell to form protective cysts containing hundreds of eggs. The golden and white cyst nematodes are major potato pests globally.",
            "Globodera rostochiensis (golden cyst nematode) and G. pallida (white cyst nematode) are quarantine pests that severely reduce potato yields. They extract nutrients from root cells, causing stunting and yield loss.",
            "Cyst nematodes have a complex life cycle involving egg hatching, juvenile penetration of roots, feeding site establishment, and cyst formation. Each generation takes about 6-8 weeks under favorable conditions.",
            "These nematodes cause characteristic symptoms including patchy yellowing, stunted growth, and poor tuber formation. Heavy infestations can reduce yields by 30-80% in susceptible potato varieties."
        ],
        "reasons": [
            "Contaminated seed tubers, soil, and farm equipment spread nematodes to new areas. Once established, populations build up over successive potato crops in the absence of rotation.",
            "Continuous potato cropping or short rotations allow nematode populations to multiply rapidly. The absence of non-host crops fails to reduce viable egg populations in soil.",
            "Favorable soil conditions including proper moisture and temperature (15-25°C) promote egg hatching and juvenile activity. Sandy soils often have higher nematode mobility than clay soils.",
            "Use of susceptible potato varieties without resistance genes allows rapid population buildup. Some varieties are particularly attractive to nematode penetration and feeding.",
            "Poor field sanitation practices including movement of infested soil on equipment, boots, and tools spread nematodes within and between farms."
        ],
        "precautions": [
            "Implement long-term crop rotation with non-host crops like cereals, legumes, or brassicas for at least 4-6 years. This reduces viable egg populations in soil significantly.",
            "Use certified nematode-free seed potatoes from clean fields. Test seed lots and soil before planting to ensure absence of viable cysts and eggs.",
            "Plant resistant potato varieties carrying H1 or other resistance genes where available. Rotate different resistance genes to prevent nematode population adaptation.",
            "Practice strict sanitation by cleaning equipment, tools, and vehicles when moving between fields. Avoid moving soil from infested to clean areas.",
            "Consider soil fumigation or biofumigation with brassica crops that release natural nematicides. Steam sterilization of small areas may be effective for high-value crops."
        ],
        "pesticides": [
            "Soil nematicides: Oxamyl (Vydate) applied as soil treatment at planting at 3-5kg per hectare. Fenamiphos (Nemacur) granules incorporated before planting provide season-long control.",
            "Biological control: Paecilomyces lilacinus fungal preparations applied to soil reduce nematode egg viability. Bacillus firmus-based products (Votivo) colonize roots and suppress nematodes.",
            "Plant extracts: Neem cake incorporation at 250kg per hectare before planting provides some nematode suppression while improving soil organic matter.",
            "Soil amendments: Chitosan applications stimulate plant defense responses against nematodes. Calcium cyanamide soil treatment provides both fertilizer and nematicide effects.",
            "Fumigants: Metam sodium soil fumigation before planting in severely infested fields. Dazomet (Basamid) granular fumigant incorporation provides broad-spectrum soil sterilization."
        ]
    },
    "Potato Virus": {
        "descriptions": [
            "Potato viruses comprise a complex group of plant pathogens including Potato Virus Y (PVY), Potato Leaf Roll Virus (PLRV), and Potato Virus X (PVX). These viruses cause various symptoms and significantly reduce tuber quality and yield.",
            "Viral infections in potatoes often result in mosaic patterns, leaf curling, stunting, and internal tuber defects. Some viruses like PLRV cause net necrosis in tubers, making them unmarketable for fresh consumption.",
            "Multiple virus species can infect potatoes simultaneously, creating complex symptom expressions and severe yield losses. Virus interactions often result in synergistic effects that are more damaging than individual infections.",
            "Potato viruses are primarily transmitted by insect vectors, particularly aphids, though some spread through mechanical transmission and infected seed tubers. Once infected, plants remain infected throughout their life cycle.",
            "Different potato viruses have varying impacts on plant physiology, from reducing photosynthesis efficiency to disrupting carbohydrate transport and storage in tubers. Early infection generally causes more severe symptoms and losses."
        ],
        "reasons": [
            "Infected seed tubers serve as the primary source of virus introduction to new crops. Vegetative propagation of potatoes ensures virus transmission from generation to generation.",
            "Aphid vectors, particularly green peach aphid and potato aphid, acquire viruses from infected plants and transmit them to healthy ones. Even brief feeding can result in virus transmission.",
            "Volunteer potatoes and infected weed hosts maintain virus reservoirs in the environment. These sources provide inoculum for aphid vectors to acquire and spread viruses.",
            "Mechanical transmission occurs through contaminated cutting tools, farm equipment, and human activities. Viruses can spread through plant wounds created during cultivation practices.",
            "Climate change and altered weather patterns affect aphid population dynamics and flight patterns, potentially increasing virus spread rates and expanding affected geographic regions."
        ],
        "precautions": [
            "Use certified virus-free seed potatoes from reputable seed certification programs. Test seed lots for major viruses before planting to ensure clean planting material.",
            "Implement strict roguing programs to remove virus-infected plants immediately upon detection. Early removal prevents virus spread to neighboring plants and reduces aphid acquisition sources.",
            "Control aphid vectors through integrated pest management including reflective mulches, insecticidal sprays, and beneficial insect conservation. Monitor aphid flights using yellow sticky traps.",
            "Maintain isolation distances between seed and commercial potato fields. Remove volunteer potatoes and weed hosts that can harbor viruses near production areas.",
            "Practice tool and equipment sanitation between plants and fields. Use bleach solutions (10%) to disinfect cutting tools and avoid mechanical virus transmission."
        ],
        "pesticides": [
            "Aphid control: Imidacloprid (Confidor) seed treatment or soil application provides early season protection. Thiamethoxam (Actara) at 0.3g per liter controls virus-transmitting aphids effectively.",
            "Systemic insecticides: Clothianidin (Dantotsu) applied as soil drench provides long-lasting aphid control. Acetamiprid (Pridee) foliar sprays target flying aphid populations.",
            "Reflective mulches: Silver reflective plastic mulches repel aphids and reduce virus transmission rates. These work particularly well in combination with chemical control measures.",
            "Mineral oils: Light horticultural oils at 10ml per liter can interfere with aphid feeding behavior and reduce virus transmission efficiency. Apply during periods of high aphid activity.",
            "Resistance inducers: While not pesticides per se, acibenzolar-S-methyl (Actigard) applications can enhance plant resistance to virus infections by activating defense pathways."
        ]
    },
    'Tomato_Bacterial_spot': {
        "descriptions": [
            "Bacterial spot is a serious disease caused by Xanthomonas species, leading to small, dark spots on tomato leaves, stems, and fruits. It thrives in warm, wet conditions and can defoliate plants.",
            "This bacterial infection manifests as angular water-soaked lesions that turn necrotic. It's a major issue in tomato production, especially in humid regions.",
            "Caused by Xanthomonas campestris pv. vesicatoria, bacterial spot affects all above-ground parts of the tomato plant, reducing yield and fruit quality.",
            "The disease spreads through rain splash and contaminated equipment. Symptoms include raised scabby spots on fruits making them unmarketable.",
            "Bacterial spot can lead to severe defoliation under favorable conditions, weakening the plant and exposing fruits to sunscald."
        ],
        "reasons": [
            "Warm temperatures (24-30°C) with high humidity and rainfall promote bacterial multiplication and spread. Overhead irrigation exacerbates the issue.",
            "Infected seeds or transplants introduce the pathogen to new areas. Poor sanitation allows buildup in fields.",
            "Dense planting reduces air circulation, creating moist microclimates favorable for bacteria.",
            "Nutrient imbalances, especially low calcium, can increase susceptibility. Stressed plants are more vulnerable.",
            "Wind-driven rain splashes bacteria from infected to healthy plants, rapidly spreading the disease."
        ],
        "precautions": [
            "Use disease-free seeds and transplants. Treat seeds with hot water to kill bacteria.",
            "Practice crop rotation with non-solanaceous crops for 2-3 years. Remove plant debris after harvest.",
            "Avoid overhead irrigation; use drip systems to keep foliage dry. Improve air circulation with proper spacing.",
            "Apply copper-based bactericides preventively during wet periods. Monitor weather for application timing.",
            "Plant resistant varieties when available. Maintain balanced nutrition to boost plant resistance."
        ],
        "pesticides": [
            "Copper-based: Kocide (copper hydroxide) at 2-3g per liter, apply every 7-10 days during wet weather.",
            "Systemic: Streptomycin sulfate at 200ppm for seed treatment or early foliar sprays.",
            "Combination: Copper + mancozeb (e.g., ManKocide) at 2.5g per liter for enhanced protection.",
            "Biological: Bacillus subtilis products like Serenade at 2-4ml per liter to suppress bacterial growth.",
            "Organic: Neem oil mixed with copper soaps for eco-friendly control in organic systems."
        ]
    },
    'Tomato_Early_blight': {
        "descriptions": [  # Copied and adapted from "Early Blight"
            "Early blight is a destructive fungal disease caused by Alternaria solani that primarily affects tomato plants. This pathogen thrives in warm, humid conditions and can significantly reduce crop yields.",
            "This fungal infection, scientifically known as Alternaria solani, manifests as distinctive dark brown spots with concentric rings on tomato leaves. It's one of the most common diseases affecting tomato crops worldwide.",
            "Early blight disease is characterized by its target-like lesions that appear on older tomato leaves first. The fungal pathogen Alternaria solani spreads rapidly during periods of high humidity and moderate temperatures.",
            "Caused by the fungus Alternaria solani, early blight creates characteristic bull's-eye patterns on tomato leaves. This disease can cause severe defoliation and reduce photosynthetic capacity of the plant.",
            "Early blight is a necrotrophic fungal disease that affects the foliage, stems, and fruits of tomato plants. The pathogen Alternaria solani produces toxins that kill plant tissue, creating the distinctive lesions."
        ],
        "reasons": [  # Similar
            "High humidity levels (above 90%) combined with temperatures between 24-29°C create ideal conditions for spore germination and infection. Poor air circulation and dense plant canopies exacerbate the problem.",
            "Prolonged leaf wetness from dew, rain, or overhead irrigation provides the moisture needed for fungal spore development. Stressed plants due to nutrient deficiencies are more susceptible to infection.",
            "The fungus overwinters in infected plant debris and soil, releasing spores during favorable weather conditions. Wind and rain splash spread the spores to healthy plants, initiating new infection cycles.",
            "Excessive nitrogen fertilization can make plants more susceptible by promoting lush growth with tender tissues. Potassium deficiency also weakens plant resistance to fungal pathogens.",
            "Poor crop rotation practices allow the pathogen to build up in soil. Growing susceptible crops in the same field repeatedly increases disease pressure and reduces natural soil suppressiveness."
        ],
        "precautions": [  # Similar
            "Implement proper crop rotation with non-solanaceous crops for at least 2-3 years. Remove and destroy infected plant debris immediately after harvest to reduce inoculum sources.",
            "Ensure adequate plant spacing (45-60cm apart) to improve air circulation and reduce humidity around plants. Avoid overhead irrigation and water plants at soil level during early morning hours.",
            "Apply preventive fungicide sprays containing copper compounds or chlorothalonil before disease symptoms appear. Begin applications when environmental conditions favor disease development.",
            "Use certified disease-free seeds and resistant varieties when available. Maintain balanced nutrition with adequate potassium and avoid excessive nitrogen fertilization that promotes susceptible growth.",
            "Install drip irrigation systems to minimize leaf wetness duration. Stake or cage plants properly to improve air circulation and prevent soil splash onto lower leaves during rainfall."
        ],
        "pesticides": [  # Similar
            "Systemic fungicides: Propiconazole (Tilt) at 0.1% concentration, apply every 10-14 days. Azoxystrobin (Quadris) at 250-500ml per hectare provides excellent protection against early blight.",
            "Contact fungicides: Mancozeb (Dithane M-45) at 2-2.5g per liter of water. Copper oxychloride (Blitox) at 3g per liter provides good preventive control when applied regularly.",
            "Combination products: Ridomil Gold MZ (metalaxyl + mancozeb) at 2.5g per liter offers both preventive and curative action. Apply at 7-10 day intervals during favorable conditions.",
            "Biological control: Bacillus subtilis-based products like Serenade at 2-4ml per liter. Trichoderma harzianum formulations can suppress fungal growth when applied as soil treatment.",
            "Organic options: Neem oil at 3-5ml per liter combined with potassium bicarbonate at 1g per liter. Bordeaux mixture (copper sulfate + lime) at 1% concentration for organic farming systems."
        ]
    },
    'Tomato_Late_blight': {
        "descriptions": [  # Copied and adapted from "Late Blight"
            "Late blight is a devastating oomycete disease caused by Phytophthora infestans, capable of destroying entire tomato crops within days under favorable conditions.",
            "This aggressive plant pathogen, Phytophthora infestans, causes water-soaked lesions that rapidly expand on tomato leaves and stems. The disease is particularly destructive during cool, moist weather.",
            "Late blight manifests as dark, irregular patches on tomato leaves with white fuzzy growth on the undersides. The pathogen is an oomycete requiring specific management.",
            "Phytophthora infestans produces sporangia that spread rapidly through wind and rain, affecting all parts of the tomato plant including fruits.",
            "Late blight disease progresses rapidly from initial infection to complete plant death within 7-10 days in tomatoes. Thrives in cool temperatures (15-25°C) with high humidity."
        ],
        "reasons": [  # Similar
            "Cool temperatures (15-25°C) combined with high humidity (>90%) and frequent rainfall create perfect conditions for late blight development. Extended leaf wetness periods allow sporangial germination.",
            "The pathogen survives in infected seed tubers and volunteer plants, providing primary inoculum for new infections. Wind-dispersed sporangia can travel several kilometers to infect new fields.",
            "Dense plant canopies with poor air circulation maintain high humidity microclimates favorable for disease development. Overhead irrigation systems prolong leaf wetness and facilitate spore movement.",
            "Late planting during monsoon seasons increases disease risk due to conducive weather conditions. Fields located in valleys or areas with poor drainage are particularly susceptible.",
            "Use of susceptible varieties without resistance genes makes crops vulnerable to infection. Inadequate crop rotation allows pathogen buildup in soil and crop debris."
        ],
        "precautions": [  # Similar
            "Plant certified disease-free transplants from reputable sources. Avoid using infected material.",
            "Implement strict field sanitation by destroying infected plant debris immediately. Remove volunteer tomato plants.",
            "Apply preventive fungicide sprays before disease symptoms appear, especially during favorable weather conditions. Use weather-based forecasting.",
            "Improve field drainage and avoid overhead irrigation. Plant in well-ventilated areas to reduce humidity.",
            "Choose resistant varieties and practice crop rotation with non-host crops for at least 2 years. Monitor fields regularly."
        ],
        "pesticides": [  # Similar
            "Systemic fungicides: Metalaxyl-M + Mancozeb (Ridomil Gold MZ) at 2.5g per liter provides excellent control. Apply at 7-day intervals during disease-favorable conditions.",
            "Contact fungicides: Propineb (Antracol) at 2.5g per liter or Metiram (Polyram) at 2g per liter. These provide protective barriers against spore germination on leaf surfaces.",
            "Newer chemistries: Cyazofamid (Ranman) at 1ml per liter offers specific activity against oomycetes. Mandipropamid (Revus) at 1.25ml per liter provides excellent late blight control.",
            "Combination treatments: Cymoxanil + Famoxadone (Equation Pro) at 2ml per liter combines contact and systemic activity. Dimethomorph + Ametoctradin (Lenko) for resistance management.",
            "Emergency control: Copper compounds like copper hydroxide at 3g per liter can slow disease progress. Fosetyl-Al (Aliette) at 2.5g per liter stimulates plant defense mechanisms."
        ]
    },
    'Tomato_Leaf_Mold': {
        "descriptions": [
            "Leaf mold is a fungal disease caused by Passalora fulva, affecting tomato leaves with yellow spots on upper surfaces and olive-green mold on undersides.",
            "This disease thrives in high humidity greenhouses, leading to defoliation and reduced yield in tomato crops.",
            "Passalora fulva infects tomato foliage, causing chlorotic spots that coalesce, severely impacting photosynthesis.",
            "Common in protected cultivation, leaf mold spreads via spores and can rapidly epidemic in humid conditions.",
            "Symptoms include powdery fungal growth on leaf undersides, leading to premature leaf drop in tomatoes."
        ],
        "reasons": [
            "High relative humidity (>85%) and moderate temperatures (20-25°C) favor spore production and infection.",
            "Poor ventilation in greenhouses or dense canopies traps moisture, promoting disease development.",
            "Overhead watering keeps leaves wet, facilitating fungal germination. Crowded plants aid spread.",
            "Infected plant debris or contaminated structures harbor the pathogen between seasons.",
            "Susceptible varieties in enclosed environments increase risk of outbreaks."
        ],
        "precautions": [
            "Improve ventilation and air circulation in greenhouses. Use fans to reduce humidity.",
            "Avoid overhead irrigation; water at base. Prune plants to open canopy.",
            "Use resistant tomato varieties. Remove lower leaves to improve airflow.",
            "Sanitize greenhouse structures and tools. Remove infected leaves promptly.",
            "Monitor humidity and temperature; use dehumidifiers if necessary."
        ],
        "pesticides": [
            "Fungicides: Chlorothalonil (Bravo) at 2ml per liter, apply preventively every 7-10 days.",
            "Systemic: Trifloxystrobin (Flint) at 1ml per liter for curative action.",
            "Biological: Trichoderma-based products for soil and foliar application to suppress fungus.",
            "Combination: Azoxystrobin + difenoconazole (Amistar Top) at 1ml per liter.",
            "Organic: Sulfur-based fungicides or potassium bicarbonate sprays for mild control."
        ]
    },
    'Tomato_Septoria_leaf_spot': {
        "descriptions": [
            "Septoria leaf spot is caused by Septoria lycopersici, producing small circular spots with dark borders on tomato leaves.",
            "This fungal disease leads to defoliation starting from lower leaves, reducing tomato plant vigor and yield.",
            "Spots have gray centers with tiny black fruiting bodies; severe cases cause complete leaf loss.",
            "Common in wet seasons, Septoria spreads via rain splash and overwinters in debris.",
            "Affects primarily foliage but can impact stems; fruits rarely directly affected but yield drops."
        ],
        "reasons": [
            "Wet weather and high humidity promote spore germination. Temperatures 20-25°C optimal.",
            "Infected plant residues in soil release spores in spring. Rain splash spreads to lower leaves.",
            "Dense planting limits air flow, maintaining moisture on leaves.",
            "Overhead irrigation prolongs leaf wetness. Nutrient-deficient plants more susceptible.",
            "Lack of crop rotation allows pathogen buildup in soil."
        ],
        "precautions": [
            "Rotate crops with non-hosts for 2-3 years. Remove and destroy infected debris.",
            "Stake plants and prune to improve air circulation. Use mulch to prevent splash.",
            "Avoid overhead watering; irrigate at base in morning.",
            "Plant resistant varieties. Space plants adequately.",
            "Monitor lower leaves for early spots and remove promptly."
        ],
        "pesticides": [
            "Contact fungicides: Mancozeb at 2.5g per liter, apply every 7-14 days.",
            "Systemic: Azoxystrobin (Quadris) at 1ml per liter for protection.",
            "Combination: Chlorothalonil + propiconazole for broad control.",
            "Biological: Bacillus amyloliquefaciens products for suppression.",
            "Organic: Copper fungicides or neem oil mixtures."
        ]
    },
    'Tomato_Spider_mites Two-spotted_spider_mite': {
        "descriptions": [  # Adapted from "Plant Pests"
            "Two-spotted spider mites are tiny arachnids that suck sap from tomato leaves, causing stippling and bronzing.",
            "Tetranychus urticae infests undersides of leaves, producing fine webs in severe cases.",
            "Mites multiply rapidly in hot, dry conditions, leading to leaf drop and reduced yield.",
            "Damage appears as yellow speckles; heavy infestations cause leaves to dry and fall.",
            "Common greenhouse pest on tomatoes, mites can develop resistance to pesticides."
        ],
        "reasons": [  # Similar to pests
            "Hot, dry weather accelerates reproduction. Low humidity favors mite populations.",
            "Overuse of broad-spectrum insecticides kills natural predators, leading to outbreaks.",
            "Dust on leaves provides habitat. Stressed plants from drought more susceptible.",
            "Infested transplants introduce mites. Crowded conditions aid spread.",
            "High nitrogen fertilization promotes succulent growth attractive to mites."
        ],
        "precautions": [  # Similar
            "Monitor undersides of leaves regularly with magnifying glass. Use sticky traps.",
            "Maintain humidity above 50% in greenhouses. Avoid dust buildup.",
            "Encourage natural predators like ladybugs and predatory mites.",
            "Use strong water sprays to dislodge mites from leaves.",
            "Isolate new plants and quarantine if infested."
        ],
        "pesticides": [
            "Miticides: Abamectin (Avid) at 0.5ml per liter, apply to undersides.",
            "Selective: Spiromesifen (Oberon) at 1ml per liter targets mites.",
            "Biological: Phytoseiulus persimilis predatory mites for release.",
            "Natural: Insecticidal soaps or neem oil at 5ml per liter.",
            "Systemic: Hexythiazox (Hexygon) for egg and nymph control."
        ]
    },
    'Tomato_Target_Spot': {
        "descriptions": [
            "Target spot is caused by Corynespora cassiicola, producing concentric ring spots on tomato leaves and fruits.",
            "This fungal disease affects foliage, stems, and fruits, leading to premature defoliation.",
            "Spots start small and expand with bull's-eye appearance, similar to early blight but distinct.",
            "Thrives in warm, humid conditions; can cause significant fruit loss in tomatoes.",
            "Pathogen survives in debris; spreads via wind and rain."
        ],
        "reasons": [
            "Warm temperatures (25-30°C) with high humidity favor infection.",
            "Infected plant residues provide inoculum. Rain splash spreads spores.",
            "Poor air circulation in dense plantings promotes disease.",
            "Overhead irrigation keeps plants wet. Stressed plants more vulnerable.",
            "Lack of rotation with non-host crops builds pathogen levels."
        ],
        "precautions": [
            "Rotate crops for 2-3 years. Remove debris after harvest.",
            "Improve spacing and pruning for better airflow.",
            "Use drip irrigation to avoid wetting foliage.",
            "Plant resistant varieties if available.",
            "Scout regularly and remove infected parts."
        ],
        "pesticides": [
            "Fungicides: Azoxystrobin (Quadris) at 1ml per liter, rotate with others.",
            "Contact: Chlorothalonil at 2ml per liter for protection.",
            "Systemic: Difenoconazole (Score) at 1ml per liter.",
            "Biological: Trichoderma viride formulations.",
            "Organic: Copper-based sprays every 7 days."
        ]
    },
    'Tomato_Tomato_Yellow_Leaf_Curl_Virus': {
        "descriptions": [  # Adapted from "Potato Virus"
            "Tomato Yellow Leaf Curl Virus (TYLCV) is a geminivirus causing leaf curling, yellowing, and stunting in tomatoes.",
            "Transmitted by whiteflies, TYLCV leads to severe yield losses, up to 100% in early infections.",
            "Symptoms include upward leaf curl, reduced leaf size, and flower drop.",
            "One of the most damaging tomato viruses worldwide, especially in tropical regions.",
            "Infected plants produce few or no fruits; virus persists in weed hosts."
        ],
        "reasons": [  # Similar to virus
            "Whitefly vectors transmit the virus from infected to healthy plants.",
            "Infected transplants or weed reservoirs maintain the virus.",
            "High whitefly populations in warm weather accelerate spread.",
            "Lack of vector control leads to epidemics.",
            "Global trade introduces new strains to areas."
        ],
        "precautions": [  # Similar
            "Use virus-free transplants. Screen greenhouses against whiteflies.",
            "Control whiteflies with traps and barriers. Remove weed hosts.",
            "Plant resistant varieties. Use reflective mulches to repel vectors.",
            "Rogue infected plants immediately.",
            "Monitor whitefly populations weekly."
        ],
        "pesticides": [
            "Insecticides for vectors: Imidacloprid (Confidor) at 0.5ml per liter.",
            "Systemic: Thiamethoxam (Actara) soil drench for whitefly control.",
            "Biological: Encarsia formosa parasitoids for whiteflies.",
            "Natural: Neem oil to deter whiteflies.",
            "Oils: Horticultural oils to smother insects."
        ]
    },
    'Tomato_Tomato_mosaic_virus': {
        "descriptions": [  # Adapted from virus
            "Tomato Mosaic Virus (ToMV) causes mosaic patterns, leaf distortion, and reduced yield in tomatoes.",
            "Mechanically transmitted virus, stable and easily spread via tools and hands.",
            "Symptoms include mottling, fern-like leaves, and fruit distortion.",
            "Persistent in seeds and debris; affects greenhouse and field tomatoes.",
            "Can reduce yield by 20-30%; interacts with other viruses for worse symptoms."
        ],
        "reasons": [  # Similar
            "Mechanical transmission through sap on tools, hands, or clothing.",
            "Infected seeds or transplants introduce the virus.",
            "Greenhouse environments favor persistence due to close plant contact.",
            "Smokers can spread via tobacco products containing the virus.",
            "Lack of sanitation between handling plants."
        ],
        "precautions": [  # Similar
            "Use certified virus-free seeds. Heat-treat seeds if necessary.",
            "Sanitize tools with bleach or milk solutions between plants.",
            "Avoid tobacco use near plants. Wash hands thoroughly.",
            "Rogue infected plants. Use resistant varieties.",
            "Isolate new plants before introducing to field."
        ],
        "pesticides": [
            "No direct antivirals; control via sanitation.",
            "Milk sprays (20% non-fat) can inactivate virus on surfaces.",
            "Disinfectants: 10% bleach for tools.",
            "Biological: None specific, focus on prevention.",
            "Organic: Essential oils for tool cleaning."
        ]
    },
    'Tomato_healthy': {
        "descriptions": [  # Adapted from "Healthy"
            "Excellent news! Your tomato plant appears to be in optimal health with no visible signs of disease or pest damage. The foliage shows vibrant green coloration and proper leaf structure.",
            "Congratulations! The analysis indicates your tomato crop is thriving with no detectable diseases. Healthy tomato plants exhibit strong cellular structure and absence of pathological symptoms.",
            "Your tomato plant sample shows all indicators of robust health - proper leaf color, intact cell structure, and no evidence of pathogenic organisms.",
            "The diagnostic results confirm your tomato plant is disease-free and maintaining excellent physiological condition.",
            "Perfect! Your tomato crop shows optimal health parameters with no signs of infections. The plant tissue appears strong and resilient."
        ],
        "reasons": [  # Similar
            "Proper cultural practices including balanced nutrition, appropriate irrigation, and good field hygiene have maintained tomato plant health.",
            "Favorable environmental conditions with moderate temperatures, adequate moisture, and good air circulation support tomato health.",
            "Use of disease-resistant varieties combined with integrated pest management has protected the tomato crop.",
            "Timely preventive treatments and optimal growing conditions have kept the tomato plant healthy.",
            "Effective farm management including clean cultivation and regular inspection has prevented disease in tomatoes."
        ],
        "precautions": [  # Similar
            "Continue current management practices to maintain tomato plant health. Regular monitoring remains crucial.",
            "Maintain preventive fungicide schedule during favorable weather for tomatoes.",
            "Practice good hygiene including tool sanitization and debris removal for tomato crops.",
            "Continue balanced fertilization avoiding excesses that predispose tomatoes to diseases.",
            "Monitor weather and prepare protective measures for high-risk periods in tomato fields."
        ],
        "pesticides": [  # Similar
            "Preventive: Light copper sprays at 15-day intervals during humid periods for tomatoes.",
            "Biological: Trichoderma or Bacillus subtilis monthly to enhance tomato health.",
            "Nutrient: Potassium phosphonate foliar to boost tomato immunity.",
            "Organic: Neem oil every 2 weeks for broad protection in tomatoes.",
            "Systemic: Light propiconazole during critical stages for tomato protection."
        ]
    }
}

def get_disease_info(disease_name):
    """Get random disease information for the detected disease"""
    if disease_name not in DISEASE_INFO:
        return {
            'description': f"Information about {disease_name} is being updated.",
            'reason': "Multiple factors may contribute to this condition.",
            'precaution': "Follow general good agricultural practices.",
            'pesticide': "Consult local agricultural extension services for recommendations."
        }
    
    disease_data = DISEASE_INFO[disease_name]
    
    return {
        'description': random.choice(disease_data['descriptions']),
        'reason': random.choice(disease_data['reasons']), 
        'precaution': random.choice(disease_data['precautions']),
        'pesticide': random.choice(disease_data['pesticides'])
    }
# ----------------- DB SETUP -----------------
def init_db():
    with sqlite3.connect("agroai.db") as conn:
        c = conn.cursor()
        # Users table
        c.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                password TEXT NOT NULL
            )
        """)
        # Irrigation plans table
        c.execute("""
            CREATE TABLE IF NOT EXISTS irrigation_plans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                crop_type TEXT,
                field_size REAL,
                soil_type TEXT,
                growth_stage TEXT,
                temperature REAL,
                humidity REAL,
                liters_per_day REAL,
                times_per_week INTEGER,
                minutes_per_session INTEGER,
                best_time TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        # Predictions table
        c.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                crop_type TEXT,
                predicted_class TEXT,
                confidence REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()

init_db()

# ----------------- HELPERS -----------------
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def transcribe_audio_groq(filepath):
    try:
        with open(filepath, "rb") as f:
            response = groq_client.audio.transcriptions.create(
                model="whisper-large-v3-turbo",
                file=f,
            )
            return response.text
    except Exception as e:
        print(f"Error in transcription: {e}")
        return "Error transcribing audio."

def get_answer_groq(question):
    try:
        response = groq_client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {"role": "system", "content": "You are a helpful agriculture chatbot for Indian farmers."},
                {"role": "user", "content": question}
            ],
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error in getting answer: {e}")
        return "Error generating answer."

def clean_text_for_audio(text):
    # Remove markdown symbols for clearer TTS
    return re.sub(r'[*#`_\-|>\n]+', ' ', text).strip()

def text_to_audio(text, filename):
    try:
        cleaned_text = clean_text_for_audio(text)
        tts = gTTS(cleaned_text)
        tts.save(f"static/audio/{filename}.mp3")
    except Exception as e:
        print(f"Error in text-to-audio: {e}")

# Grad-CAM function for disease detection visualization
def generate_gradcam(model, input_tensor, target_layer, target_class=None):
    model.eval()
    gradients = []
    activations = []

    def forward_hook(module, input, output):
        activations.append(output.detach())

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0].detach())

    handle_fw = target_layer.register_forward_hook(forward_hook)
    handle_bw = target_layer.register_backward_hook(backward_hook)

    output = model(input_tensor)
    if target_class is None:
        target_class = output.argmax(dim=1).item()

    loss = output[0, target_class]
    model.zero_grad()
    loss.backward()

    grad = gradients[0][0]  # [C, H, W]
    act = activations[0][0]  # [C, H, W]

    weights = grad.mean(dim=(1, 2))  # Global Average Pooling
    cam = torch.zeros(act.shape[1:], dtype=torch.float32).to(DEVICE)
    for i, w in enumerate(weights):
        cam += w * act[i]

    cam = F.relu(cam)
    cam = cam.cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)  # Normalize 0-1

    handle_fw.remove()
    handle_bw.remove()
    return cam, target_class

# ----------------- ROUTES -----------------
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/details')
def details():
    return render_template('detail.html')

@app.route('/contact')
def contact():
    return render_template('contact_Final.html')

@app.route('/weather')
def weather():
    if "user_id" not in session:
        flash("Please log in to access the weather page.", "danger")
        return redirect(url_for('login_signup'))
    return render_template('weather.html', username=session.get("username"))

# ----------------- DISEASE DETECTION -----------------
@app.route('/detect', methods=['POST'])
def detect():
    if "user_id" not in session:
        flash("Please log in to use disease detection.", "danger")
        return redirect(url_for("login_signup"))

    file = request.files.get("file")
    crop_type = request.form.get("crop")

    if not file or not crop_type:
        flash("Please upload an image and select crop.", "danger")
        return redirect(url_for("dashboard"))

    if crop_type not in ['Tomato', 'Potato']:
        flash("Disease detection currently available only for Tomato and Potato.", "danger")
        return redirect(url_for("dashboard"))

    model = tomato_model if crop_type == 'Tomato' else potato_model
    class_names = tomato_class_names if crop_type == 'Tomato' else potato_class_names

    if model is None:
        flash(f"{crop_type} disease detection model is not available.", "danger")
        return redirect(url_for("dashboard"))

    try:
        # Read and transform image
        image = Image.open(file).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(DEVICE)

        # Find last convolutional layer dynamically
        target_layer = None
        for name, module in reversed(list(model.named_modules())):
            if isinstance(module, torch.nn.Conv2d):
                target_layer = module
                break

        if target_layer is None:
            flash("Could not find convolutional layer for Grad-CAM.", "danger")
            return redirect(url_for("dashboard"))

        # Generate Grad-CAM and prediction
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            confidence, pred = torch.max(probabilities, 1)
            class_idx = pred.item()
            confidence = confidence.item() * 100  # Convert to percentage

        cam, pred_idx = generate_gradcam(model, input_tensor, target_layer)
        predicted_class = class_names[class_idx]
        display_class = predicted_class.replace('_', ' ')

        # Get disease information
        disease_info = get_disease_info(predicted_class)

        # Resize image and CAM
        img_resized = image.resize((224, 224))
        img_np = np.array(img_resized)

        # Overlay heatmap
        if 'healthy' in predicted_class.lower():
            overlay = cv2.addWeighted(img_np, 0.7, np.full_like(img_np, (0, 255, 0)), 0.3, 0)
        else:
            cam_uint8 = np.uint8(255 * cam)
            cam_resized = cv2.resize(cam_uint8, (img_np.shape[1], img_np.shape[0]))
            cmap = CLASS_COLORMAPS.get(predicted_class, cv2.COLORMAP_JET)
            heatmap = cv2.applyColorMap(cam_resized, cmap)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)

        # Convert overlay to PIL Image
        result_img = Image.fromarray(overlay)

        # Convert to base64 for HTML rendering
        buffered = io.BytesIO()
        result_img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        # Store prediction in database
        with sqlite3.connect("agroai.db") as conn:
            c = conn.cursor()
            c.execute("""
                INSERT INTO predictions (user_id, crop_type, predicted_class, confidence)
                VALUES (?, ?, ?, ?)
            """, (session["user_id"], crop_type, predicted_class, confidence))
            conn.commit()

        # Fetch recent predictions
        with sqlite3.connect("agroai.db") as conn:
            c = conn.cursor()
            c.execute("""
                SELECT created_at, crop_type, predicted_class, confidence
                FROM predictions
                WHERE user_id = ?
                ORDER BY created_at DESC
                LIMIT 5
            """, (session["user_id"],))
            recent_predictions = c.fetchall()
        # Get disease information
        print(f"Predicted Class: {predicted_class}, Disease Info: {disease_info}")

        return render_template("dashboard.html", 
                     username=session.get("username"), 
                     result_image=img_str, 
                     predicted_class=display_class,
                     confidence=f"{confidence:.2f}%",
                     recent_predictions=recent_predictions,
                     chart_data={"labels": ["Healthy", "Diseased"], "values": [35, 7]},
                     disease_info=disease_info)  # Add disease_info to template variables

    except Exception as e:
        print(f"Error in disease detection: {e}")
        flash("Error processing image. Please try again.", "danger")
        return redirect(url_for("dashboard"))
# ----------------- IRRIGATION -----------------
@app.route('/irrigation', methods=['GET', 'POST'])
def irrigation():
    if "user_id" not in session:
        flash("Please log in to access the irrigation planner.", "danger")
        return redirect(url_for('login_signup'))

    if request.method == "POST":
        crop_type = request.form.get("crop-type")
        field_size = request.form.get("field-size")
        soil_type = request.form.get("soil-type")
        growth_stage = request.form.get("growth-stage")
        temperature = request.form.get("temperature")
        humidity = request.form.get("humidity")

        if not all([crop_type, field_size, soil_type, growth_stage, temperature, humidity]):
            flash("Please fill in all fields.", "danger")
            return redirect(url_for("irrigation"))

        try:
            field_size = float(field_size)
            temperature = float(temperature)
            humidity = float(humidity)
        except ValueError:
            flash("Field size, temperature, and humidity must be numbers.", "danger")
            return redirect(url_for("irrigation"))

        # Water calculation
        crop_water_needs = {'tomato': 600, 'potato': 500, 'wheat': 450, 'rice': 1200, 'maize': 550, 'cotton': 700}
        soil_factor = {'clay': 0.8, 'sandy': 1.2, 'loamy': 1.0, 'silt': 0.9}
        growth_factor = {'seedling': 0.6, 'vegetative': 0.8, 'flowering': 1.0, 'fruiting': 0.9}

        water_need = crop_water_needs.get(crop_type, 500) * field_size * \
                     soil_factor.get(soil_type, 1.0) * growth_factor.get(growth_stage, 1.0)
        if temperature > 30:
            water_need *= 1.2
        if humidity < 50:
            water_need *= 1.1

        times_per_week = 4 if temperature > 25 else 3
        minutes_per_session = round(water_need / (times_per_week * 10))
        best_time = 'Evening (6-8 PM)' if temperature > 25 else 'Morning (5-8 AM)'

        # Store in database
        with sqlite3.connect("agroai.db") as conn:
            c = conn.cursor()
            c.execute("""
                INSERT INTO irrigation_plans (user_id, crop_type, field_size, soil_type, growth_stage, temperature, humidity, liters_per_day, times_per_week, minutes_per_session, best_time)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (session["user_id"], crop_type, field_size, soil_type, growth_stage, temperature, humidity, round(water_need), times_per_week, minutes_per_session, best_time))
            conn.commit()

        return render_template('irrigation.html', 
                             username=session.get("username"), 
                             plan={
                                 'liters_per_day': round(water_need),
                                 'times_per_week': times_per_week,
                                 'minutes_per_session': minutes_per_session,
                                 'best_time': best_time
                             })

    return render_template('irrigation.html', username=session.get("username"), plan=None)

# ----------------- LOGIN / SIGNUP -----------------
@app.route('/login_signup', methods=['GET', 'POST'])
def login_signup():
    if request.method == "POST":
        form_type = request.form.get("form_type")
        if form_type == "signup":
            username = request.form.get("username")
            email = request.form.get("email")
            password = request.form.get("password")
            confirm_password = request.form.get("confirm_password")
            if not all([username, email, password, confirm_password]):
                flash("Please fill in all fields.", "danger")
                return redirect(url_for("login_signup"))
            if password != confirm_password:
                flash("Passwords do not match.", "danger")
                return redirect(url_for("login_signup"))
            try:
                with sqlite3.connect("agroai.db") as conn:
                    c = conn.cursor()
                    c.execute("INSERT INTO users (username, email, password) VALUES (?, ?, ?)",
                              (username, email, generate_password_hash(password)))
                    conn.commit()
                flash("Account created successfully! Please log in.", "success")
            except sqlite3.IntegrityError:
                flash("Email already exists!", "danger")
            return redirect(url_for("login_signup"))

        elif form_type == "login":
            email = request.form.get("email")
            password = request.form.get("password")
            if not all([email, password]):
                flash("Please fill in all fields.", "danger")
                return redirect(url_for("login_signup"))
            with sqlite3.connect("agroai.db") as conn:
                c = conn.cursor()
                c.execute("SELECT id, username, password FROM users WHERE email=?", (email,))
                user = c.fetchone()
            if user and check_password_hash(user[2], password):
                session["user_id"] = user[0]
                session["username"] = user[1]
                flash("Login successful!", "success")
                return redirect(url_for("dashboard"))
            else:
                flash("Invalid email or password!", "danger")
                return redirect(url_for("login_signup"))
    return render_template('login_signup.html')

@app.route('/dashboard')
def dashboard():
    if "user_id" not in session:
        flash("Please log in to access the dashboard.", "danger")
        return redirect(url_for("login_signup"))
    
    # Fetch recent predictions for the dashboard
    recent_predictions = []
    try:
        with sqlite3.connect("agroai.db") as conn:
            c = conn.cursor()
            c.execute("""
                SELECT created_at, crop_type, predicted_class, confidence
                FROM predictions
                WHERE user_id = ?
                ORDER BY created_at DESC
                LIMIT 5
            """, (session["user_id"],))
            recent_predictions = c.fetchall()
    except Exception as e:
        print(f"Error fetching predictions: {e}")
    
    chart_data = {"labels": ["Healthy", "Diseased"], "values": [35, 7]}
    return render_template('dashboard.html', 
                         username=session["username"], 
                         chart_data=chart_data,
                         recent_predictions=recent_predictions)

@app.route('/logout')
def logout():
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for("home"))

# ----------------- CHATBOT -----------------
@app.route('/agroai')
def agroai():
    return render_template('chatbot.html')

@app.route('/chat', methods=['POST'])
def chat():
    answer = ""
    voice_filename = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))

    if 'audio' in request.files:
        audio = request.files['audio']
        if audio and allowed_file(audio.filename):
            filename = secure_filename(audio.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            audio.save(filepath)
            transcription = transcribe_audio_groq(filepath)
            answer = get_answer_groq(transcription)
            text_to_audio(answer, voice_filename)

    elif 'text' in request.form:
        question = request.form['text']
        answer = get_answer_groq(question)
        text_to_audio(answer, voice_filename)

    if answer:
        # Convert Markdown to HTML for frontend
        answer_html = markdown.markdown(answer, extensions=["tables"])
        return jsonify({
            'text': answer_html,
            'voice': url_for('static', filename=f'audio/{voice_filename}.mp3')
        })

    return jsonify({'text': 'No valid input found'}), 400

# ----------------- MAIN -----------------
if __name__ == '__main__':
    app.run(debug=True)