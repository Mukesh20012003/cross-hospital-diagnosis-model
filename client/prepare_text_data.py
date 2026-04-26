# client/prepare_text_data.py

"""
Clinical Text Dataset Preparation
====================================
Generates synthetic clinical notes for federated learning demo.

In a REAL hospital:
  - Clinical notes come from EHR system (Epic, Cerner)
  - Format: HL7 FHIR, plain text, structured notes
  - Types: Admission notes, discharge summaries, progress notes
  - NLP preprocessing: tokenization, stopword removal, TF-IDF

We generate synthetic clinical notes that simulate real patterns:
  - Different diseases have different keyword patterns
  - Each hospital gets its own set of clinical notes
  
5 Disease Categories:
  0: Healthy / No significant findings
  1: Cardiovascular disease
  2: Respiratory disease
  3: Neurological condition
  4: Musculoskeletal disorder
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import json


CLASS_NAMES = {
    0: "Healthy",
    1: "Cardiovascular",
    2: "Respiratory",
    3: "Neurological",
    4: "Musculoskeletal"
}

# Clinical note templates for each disease category
CLINICAL_TEMPLATES = {
    0: [  # Healthy
        "Patient presents with no significant complaints. Vitals within normal limits. Heart sounds normal. Lungs clear bilaterally. No abnormalities detected on examination.",
        "Routine checkup. Patient reports feeling well. Blood pressure normal. Heart rate regular. No chest pain or shortness of breath. Physical exam unremarkable.",
        "Annual wellness visit. Patient is in good health. No new symptoms reported. Normal vital signs. Clear lungs. Regular heart rhythm. Healthy weight maintained.",
        "Follow-up visit. Patient recovering well. No fever or pain. Appetite is good. Sleep patterns normal. Exercise tolerance good. No medication changes needed.",
        "Preventive care visit. Patient denies any complaints. Normal blood work results. BMI within healthy range. No signs of disease. Counseled on healthy lifestyle.",
        "Patient appears well nourished and in no distress. Vital signs stable. Cardiovascular exam normal. Respiratory exam clear. Neurological exam intact.",
        "Wellness examination reveals no acute findings. Patient maintains active lifestyle. Lab results within reference ranges. Continue current health maintenance plan.",
    ],
    1: [  # Cardiovascular
        "Patient presents with chest pain radiating to left arm. Elevated blood pressure at 160/95. ECG shows ST elevation. Troponin levels elevated. Suspect acute coronary syndrome.",
        "History of hypertension and hyperlipidemia. Patient reports palpitations and dizziness. Heart murmur detected on auscultation. Echocardiogram ordered. Started on beta blocker.",
        "Patient admitted with congestive heart failure exacerbation. Bilateral leg edema. Jugular venous distension noted. BNP levels significantly elevated. Started diuretic therapy.",
        "Coronary artery disease patient presenting with unstable angina. Chest tightness on exertion. Stress test positive for ischemia. Cardiology referral for catheterization.",
        "Atrial fibrillation detected on monitoring. Irregular heart rhythm. Risk of stroke assessed. Anticoagulation therapy initiated. Blood pressure medication adjusted.",
        "Patient with history of myocardial infarction reports recurrent chest discomfort. Cardiac enzymes mildly elevated. Blood pressure poorly controlled. Medication review needed.",
        "Peripheral vascular disease noted with diminished pedal pulses. Ankle brachial index reduced. Claudication symptoms present. Vascular surgery consultation recommended.",
    ],
    2: [  # Respiratory
        "Patient presents with persistent cough productive of green sputum. Fever of 101F. Chest X-ray reveals right lower lobe consolidation. Diagnosis pneumonia. Started antibiotics.",
        "Chronic obstructive pulmonary disease exacerbation. Severe dyspnea at rest. Oxygen saturation 88 percent on room air. Wheezing heard bilaterally. Nebulizer treatment initiated.",
        "Asthma patient with acute bronchospasm. Peak flow reduced to 50 percent predicted. Bilateral expiratory wheezing. Administered albuterol and corticosteroids.",
        "Suspected pulmonary embolism. Acute onset pleuritic chest pain and tachypnea. D-dimer elevated. CT pulmonary angiogram confirms bilateral pulmonary emboli. Anticoagulation started.",
        "Patient with COVID-19 pneumonia. Bilateral ground glass opacities on imaging. Oxygen requirements increasing. Inflammatory markers elevated. Dexamethasone administered.",
        "Tuberculosis screening positive. Patient reports night sweats weight loss and chronic cough. Sputum acid fast bacilli testing ordered. Chest imaging shows upper lobe infiltrates.",
        "Pleural effusion identified on chest X-ray. Patient reports progressive dyspnea. Thoracentesis performed. Fluid analysis pending. Diuretic therapy considered.",
    ],
    3: [  # Neurological
        "Patient presents with sudden onset right sided weakness and slurred speech. CT scan shows left middle cerebral artery territory infarct. Acute ischemic stroke confirmed.",
        "Chronic migraine patient with increasing frequency of headaches. Associated photophobia and nausea. Neurological exam focal. MRI brain ordered. Preventive medication started.",
        "New onset seizure in previously healthy adult. Generalized tonic clonic activity witnessed. Post-ictal confusion. EEG and brain MRI scheduled. Started antiepileptic medication.",
        "Progressive memory loss and confusion in elderly patient. Mini mental state exam score reduced. Suspect early Alzheimer disease. Neuropsychological testing ordered.",
        "Parkinson disease patient with worsening tremor and bradykinesia. Gait instability increasing fall risk. Dopamine agonist dose adjusted. Physical therapy recommended.",
        "Multiple sclerosis relapse with new onset optic neuritis. Visual acuity decreased. MRI shows new demyelinating lesions. Pulse steroid therapy initiated.",
        "Peripheral neuropathy with burning pain and numbness in feet. Nerve conduction studies show axonal damage. Vitamin B12 levels checked. Gabapentin prescribed for pain.",
    ],
    4: [  # Musculoskeletal
        "Patient presents with low back pain radiating to left leg. Positive straight leg raise test. MRI shows L4-L5 disc herniation with nerve root compression.",
        "Rheumatoid arthritis flare with bilateral hand joint swelling and morning stiffness lasting over one hour. Inflammatory markers elevated. Methotrexate dose increased.",
        "Osteoarthritis of bilateral knees with progressive pain on weight bearing. Crepitus noted on examination. X-ray shows joint space narrowing. Physical therapy ordered.",
        "Acute gout attack in right first metatarsophalangeal joint. Erythema warmth and exquisite tenderness. Uric acid levels elevated. Colchicine and anti-inflammatory prescribed.",
        "Rotator cuff tear suspected after fall. Limited shoulder range of motion with pain on abduction. MRI confirms full thickness supraspinatus tear. Orthopedic referral made.",
        "Fibromyalgia patient with widespread musculoskeletal pain and fatigue. Multiple tender points on examination. Sleep disturbance reported. Multimodal treatment plan discussed.",
        "Osteoporosis with vertebral compression fracture. Acute thoracic back pain. DEXA scan shows T-score below negative 2.5. Calcium vitamin D and bisphosphonate therapy started.",
    ]
}


def generate_clinical_notes(samples_per_class=60):
    """Generate synthetic clinical notes dataset"""
    
    notes = []
    labels = []
    
    for class_id, templates in CLINICAL_TEMPLATES.items():
        for _ in range(samples_per_class):
            # Pick a random template
            base_note = np.random.choice(templates)
            
            # Add some variation
            variations = [
                f"Patient age {np.random.randint(25, 85)}. ",
                f"Gender {'male' if np.random.random() > 0.5 else 'female'}. ",
                f"BMI {np.random.uniform(18, 35):.1f}. ",
            ]
            
            note = np.random.choice(variations) + base_note
            
            # Add random additional detail
            additional = [
                " Follow-up scheduled in two weeks.",
                " Patient educated on condition management.",
                " Labs ordered for further evaluation.",
                " Referral to specialist pending.",
                " Current medications reviewed and reconciled.",
                " Patient advised to return if symptoms worsen.",
                "",
            ]
            note += np.random.choice(additional)
            
            notes.append(note)
            labels.append(class_id)
    
    return notes, labels


def prepare_text_dataset(num_hospitals=3, samples_per_class=60):
    """Prepare clinical text dataset for federated learning"""
    
    print("\n" + "=" * 60)
    print("  📝 CLINICAL TEXT DATASET PREPARATION")
    print("  Type: Doctor's Notes / Clinical Notes")
    print("  Task: Disease Category Prediction")
    print("  Classes: Healthy, Cardiovascular, Respiratory,")
    print("           Neurological, Musculoskeletal")
    print("=" * 60)
    
    # Generate notes
    print(f"\n  📝 Generating {samples_per_class} notes per class...")
    notes, labels = generate_clinical_notes(samples_per_class)
    
    total = len(notes)
    labels = np.array(labels)
    
    print(f"  ✅ Generated {total} clinical notes")
    for cls_id, cls_name in CLASS_NAMES.items():
        count = (labels == cls_id).sum()
        print(f"     {cls_name}: {count} notes")
    
    # Create bag-of-words features
    print(f"\n  🔤 Creating bag-of-words features...")
    vectorizer = CountVectorizer(max_features=500, stop_words='english')
    X_bow = vectorizer.fit_transform(notes).toarray().astype(np.float32)
    
    vocab = vectorizer.get_feature_names_out()
    print(f"  ✅ Vocabulary size: {len(vocab)} words")
    print(f"  ✅ Feature matrix shape: {X_bow.shape}")
    
    # Shuffle
    indices = np.random.permutation(total)
    X_bow = X_bow[indices]
    labels = labels[indices]
    notes_shuffled = [notes[i] for i in indices]
    
    # Partition for hospitals
    print(f"\n  🏥 Partitioning into {num_hospitals} hospitals...")
    
    split_size = total // num_hospitals
    
    for i in range(num_hospitals):
        hospital_name = f"hospital_{i+1}"
        text_dir = os.path.join("data", hospital_name, "text")
        os.makedirs(text_dir, exist_ok=True)
        
        start = i * split_size
        end = total if i == num_hospitals - 1 else start + split_size
        
        h_X = X_bow[start:end]
        h_y = labels[start:end]
        h_notes = notes_shuffled[start:end]
        
        # Train/val split
        X_train, X_val, y_train, y_val = train_test_split(
            h_X, h_y, test_size=0.2, random_state=42, stratify=h_y
        )
        
        # Save as PyTorch tensors
        text_data = {
            "X_train": torch.FloatTensor(X_train),
            "y_train": torch.LongTensor(y_train),
            "X_val": torch.FloatTensor(X_val),
            "y_val": torch.LongTensor(y_val)
        }
        
        torch.save(text_data, os.path.join(text_dir, "text_data.pt"))
        
        # Save sample notes (for reference)
        with open(os.path.join(text_dir, "sample_notes.txt"), "w") as f:
            for j, note in enumerate(h_notes[:10]):
                f.write(f"--- Note {j+1} (Class: {CLASS_NAMES[h_y[j]]}) ---\n")
                f.write(note + "\n\n")
        
        # Save metadata
        metadata = {
            "hospital_name": hospital_name,
            "data_type": "Clinical Text (Doctor's Notes)",
            "total_notes": len(h_X),
            "train_notes": len(X_train),
            "val_notes": len(X_val),
            "vocab_size": len(vocab),
            "feature_type": "Bag-of-Words",
            "classes": CLASS_NAMES,
            "class_distribution": {
                CLASS_NAMES[c]: int((h_y == c).sum()) for c in range(5)
            },
            "source": "Synthetic clinical notes (simulating hospital EHR text)",
            "real_world_source": "Hospital EHR system (Epic, Cerner) clinical notes",
            "privacy_note": "Clinical notes remain on-premises. Only model weights are shared."
        }
        
        with open(os.path.join(text_dir, "text_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"  ✅ {hospital_name}: {len(X_train)} train, {len(X_val)} val notes")
    
    # Global test set
    test_dir = os.path.join("data", "test_set", "text")
    os.makedirs(test_dir, exist_ok=True)
    
    test_size = min(50, total // 5)
    test_data = {
        "X_test": torch.FloatTensor(X_bow[:test_size]),
        "y_test": torch.LongTensor(labels[:test_size])
    }
    torch.save(test_data, os.path.join(test_dir, "text_test_data.pt"))
    
    # Save vocabulary
    vocab_path = os.path.join("data", "vocabulary.json")
    with open(vocab_path, "w") as f:
        json.dump({"vocabulary": list(vocab), "size": len(vocab)}, f)
    
    print(f"\n  ✅ Test set: {test_size} notes saved")
    print(f"  ✅ Vocabulary saved: {vocab_path}")
    
    print(f"\n  {'=' * 50}")
    print(f"  ✅ TEXT DATASET PREPARATION COMPLETE!")
    print(f"  {'=' * 50}")
    print(f"""
  Files Created:
  ├── data/hospital_1/text/
  │   ├── text_data.pt            (Bag-of-words tensors)
  │   ├── text_metadata.json      (Class info, vocab size)
  │   └── sample_notes.txt        (Sample clinical notes)
  ├── data/hospital_2/text/
  ├── data/hospital_3/text/
  ├── data/test_set/text/
  │   └── text_test_data.pt
  └── data/vocabulary.json
  
  Model: MLP (Linear layers) — defined in server/text_model.py
  Classes: Healthy | Cardiovascular | Respiratory | Neurological | Musculoskeletal
  🔒 No clinical notes are shared — only model weights!
    """)


if __name__ == "__main__":
    prepare_text_dataset(num_hospitals=3, samples_per_class=60)