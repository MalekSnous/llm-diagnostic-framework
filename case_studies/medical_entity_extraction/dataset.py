"""
Medical entity-extraction dataset.

Each case is ``{"text", "entities", "difficulty"}`` where ``entities`` are the
canonical medical entities (conditions, medications, procedures) expected in the
model's answer. Matching is containment-based (see
``Evaluator.fuzzy_entity_metrics``), so "diabetes" matches "type 2 diabetes".

Difficulty tiers (increasing):
- easy   : explicit, full words, 1-3 entities.
- medium : full words, several entities, more clinical context.
- hard   : clinical shorthand / abbreviations the model must expand.
- expert : dense, heavily abbreviated, long or ambiguous notes.

Patient names are intentionally NOT expected entities — the task targets
clinical entities, and rewarding name extraction would reward verbatim copying.
"""

CASES = [
    # ----------------------------------------------------------------- easy
    {
        "text": "Patient presents with hypertension and type 2 diabetes. Prescribed metformin 500mg twice daily.",
        "entities": ["hypertension", "type 2 diabetes", "metformin"],
        "difficulty": "easy",
    },
    {
        "text": "32-year-old female diagnosed with asthma. Treatment includes an albuterol inhaler as needed.",
        "entities": ["asthma", "albuterol"],
        "difficulty": "easy",
    },
    {
        "text": "Patient presents with severe migraine headaches. Prescribed sumatriptan 100mg as needed.",
        "entities": ["migraine", "sumatriptan"],
        "difficulty": "easy",
    },
    {
        "text": "Diagnosed with hypothyroidism. Started on levothyroxine 50 micrograms daily.",
        "entities": ["hypothyroidism", "levothyroxine"],
        "difficulty": "easy",
    },
    {
        "text": "Patient has gastroesophageal reflux disease. Prescribed omeprazole 20mg once daily.",
        "entities": ["gastroesophageal reflux disease", "omeprazole"],
        "difficulty": "easy",
    },
    {
        "text": "Seasonal allergic rhinitis. Recommended loratadine 10mg daily during pollen season.",
        "entities": ["allergic rhinitis", "loratadine"],
        "difficulty": "easy",
    },
    {
        "text": "Patient with major depressive disorder started on sertraline 50mg daily.",
        "entities": ["major depressive disorder", "sertraline"],
        "difficulty": "easy",
    },
    {
        "text": "Bacterial sinusitis diagnosed. Prescribed amoxicillin 500mg three times daily for 10 days.",
        "entities": ["sinusitis", "amoxicillin"],
        "difficulty": "easy",
    },
    {
        "text": "Patient diagnosed with osteoporosis. Started on alendronate weekly and calcium supplements.",
        "entities": ["osteoporosis", "alendronate", "calcium"],
        "difficulty": "easy",
    },
    {
        "text": "Type 1 diabetes managed with insulin glargine at bedtime and insulin aspart with meals.",
        "entities": ["type 1 diabetes", "insulin glargine", "insulin aspart"],
        "difficulty": "easy",
    },
    {
        "text": "Patient with high cholesterol prescribed atorvastatin 20mg nightly.",
        "entities": ["hypercholesterolemia", "atorvastatin"],
        "difficulty": "easy",
    },
    {
        "text": "Urinary tract infection confirmed. Started nitrofurantoin 100mg twice daily for 5 days.",
        "entities": ["urinary tract infection", "nitrofurantoin"],
        "difficulty": "easy",
    },
    {
        "text": "Patient with epilepsy maintained on levetiracetam 500mg twice daily.",
        "entities": ["epilepsy", "levetiracetam"],
        "difficulty": "easy",
    },
    {
        "text": "Diagnosis of generalized anxiety disorder. Started escitalopram 10mg daily.",
        "entities": ["generalized anxiety disorder", "escitalopram"],
        "difficulty": "easy",
    },
    {
        "text": "Patient has rheumatoid arthritis treated with methotrexate weekly and folic acid.",
        "entities": ["rheumatoid arthritis", "methotrexate", "folic acid"],
        "difficulty": "easy",
    },
    {
        "text": "Community-acquired pneumonia. Prescribed azithromycin 500mg on day one then 250mg daily.",
        "entities": ["community-acquired pneumonia", "azithromycin"],
        "difficulty": "easy",
    },
    {
        "text": "Patient with atrial fibrillation started on apixaban 5mg twice daily for stroke prevention.",
        "entities": ["atrial fibrillation", "apixaban"],
        "difficulty": "easy",
    },
    {
        "text": "Iron deficiency anemia diagnosed. Started ferrous sulfate 325mg daily.",
        "entities": ["iron deficiency anemia", "ferrous sulfate"],
        "difficulty": "easy",
    },
    {
        "text": "Patient with chronic obstructive pulmonary disease prescribed tiotropium inhaler daily.",
        "entities": ["chronic obstructive pulmonary disease", "tiotropium"],
        "difficulty": "easy",
    },
    {
        "text": "Gout flare in the right great toe. Started colchicine and advised to continue allopurinol.",
        "entities": ["gout", "colchicine", "allopurinol"],
        "difficulty": "easy",
    },
    {
        "text": "Patient with hypertension prescribed amlodipine 5mg daily.",
        "entities": ["hypertension", "amlodipine"],
        "difficulty": "easy",
    },
    {
        "text": "Diagnosed with bacterial conjunctivitis. Prescribed erythromycin ophthalmic ointment.",
        "entities": ["conjunctivitis", "erythromycin"],
        "difficulty": "easy",
    },
    {
        "text": "Patient with insomnia. Started zolpidem 5mg at bedtime as needed.",
        "entities": ["insomnia", "zolpidem"],
        "difficulty": "easy",
    },
    {
        "text": "Hypertension and edema; prescribed hydrochlorothiazide 25mg each morning.",
        "entities": ["hypertension", "edema", "hydrochlorothiazide"],
        "difficulty": "easy",
    },
    {
        "text": "Patient diagnosed with pulmonary embolism. Started on rivaroxaban.",
        "entities": ["pulmonary embolism", "rivaroxaban"],
        "difficulty": "easy",
    },
    # --------------------------------------------------------------- medium
    {
        "text": "67-year-old admitted with acute myocardial infarction. Emergency coronary angioplasty performed and a drug-eluting stent placed.",
        "entities": ["acute myocardial infarction", "angioplasty", "stent"],
        "difficulty": "medium",
    },
    {
        "text": "28-year-old pregnant patient with gestational diabetes. Insulin therapy initiated after diet failed to control glucose.",
        "entities": ["pregnancy", "gestational diabetes", "insulin"],
        "difficulty": "medium",
    },
    {
        "text": "Patient with congestive heart failure presents with dyspnea and bilateral leg edema. Started furosemide and lisinopril; echocardiogram ordered.",
        "entities": [
            "congestive heart failure",
            "dyspnea",
            "edema",
            "furosemide",
            "lisinopril",
            "echocardiogram",
        ],
        "difficulty": "medium",
    },
    {
        "text": "Known cirrhosis with new-onset ascites. Therapeutic paracentesis performed; spironolactone initiated.",
        "entities": ["cirrhosis", "ascites", "paracentesis", "spironolactone"],
        "difficulty": "medium",
    },
    {
        "text": "Patient with chronic kidney disease stage 3 and anemia. Prescribed erythropoietin injections and referred to nephrology.",
        "entities": ["chronic kidney disease", "anemia", "erythropoietin"],
        "difficulty": "medium",
    },
    {
        "text": "Suspected appendicitis confirmed on CT. Underwent laparoscopic appendectomy; started on cefazolin perioperatively.",
        "entities": ["appendicitis", "computed tomography", "appendectomy", "cefazolin"],
        "difficulty": "medium",
    },
    {
        "text": "Newly diagnosed breast cancer. Patient scheduled for lumpectomy followed by adjuvant chemotherapy and tamoxifen.",
        "entities": ["breast cancer", "lumpectomy", "chemotherapy", "tamoxifen"],
        "difficulty": "medium",
    },
    {
        "text": "Patient presents with deep vein thrombosis of the left leg confirmed by ultrasound. Started on enoxaparin bridging to warfarin.",
        "entities": ["deep vein thrombosis", "ultrasound", "enoxaparin", "warfarin"],
        "difficulty": "medium",
    },
    {
        "text": "Type 2 diabetic with diabetic neuropathy and retinopathy. Metformin continued, gabapentin added for neuropathic pain.",
        "entities": ["type 2 diabetes", "neuropathy", "retinopathy", "metformin", "gabapentin"],
        "difficulty": "medium",
    },
    {
        "text": "Patient with Crohn's disease flare. Started on prednisone taper and maintained on azathioprine; colonoscopy scheduled.",
        "entities": ["Crohn's disease", "prednisone", "azathioprine", "colonoscopy"],
        "difficulty": "medium",
    },
    {
        "text": "Acute ischemic stroke; received tissue plasminogen activator within the window. Aspirin and atorvastatin started for secondary prevention.",
        "entities": ["ischemic stroke", "tissue plasminogen activator", "aspirin", "atorvastatin"],
        "difficulty": "medium",
    },
    {
        "text": "Patient with hyperthyroidism due to Graves disease. Started methimazole and propranolol for symptom control.",
        "entities": ["hyperthyroidism", "Graves disease", "methimazole", "propranolol"],
        "difficulty": "medium",
    },
    {
        "text": "Severe community-acquired pneumonia with sepsis. Admitted to ICU, started piperacillin-tazobactam and vasopressors.",
        "entities": ["community-acquired pneumonia", "sepsis", "piperacillin-tazobactam"],
        "difficulty": "medium",
    },
    {
        "text": "Patient with osteoarthritis of the knee. Underwent intra-articular corticosteroid injection; advised acetaminophen for pain.",
        "entities": ["osteoarthritis", "corticosteroid injection", "acetaminophen"],
        "difficulty": "medium",
    },
    {
        "text": "Newly diagnosed HIV. Started on antiretroviral therapy with tenofovir, emtricitabine, and dolutegravir.",
        "entities": ["HIV", "antiretroviral therapy", "tenofovir", "emtricitabine", "dolutegravir"],
        "difficulty": "medium",
    },
    {
        "text": "Patient with symptomatic cholelithiasis. Laparoscopic cholecystectomy performed without complication.",
        "entities": ["cholelithiasis", "cholecystectomy"],
        "difficulty": "medium",
    },
    {
        "text": "Bipolar disorder with acute mania. Admitted and stabilized on lithium and olanzapine.",
        "entities": ["bipolar disorder", "mania", "lithium", "olanzapine"],
        "difficulty": "medium",
    },
    {
        "text": "Patient with end-stage renal disease on hemodialysis. Presents with hyperkalemia; treated with calcium gluconate and insulin-dextrose.",
        "entities": [
            "end-stage renal disease",
            "hemodialysis",
            "hyperkalemia",
            "calcium gluconate",
            "insulin",
        ],
        "difficulty": "medium",
    },
    {
        "text": "Patient with prostate cancer elected radical prostatectomy. Pathology pending; PSA to be monitored.",
        "entities": ["prostate cancer", "prostatectomy", "prostate-specific antigen"],
        "difficulty": "medium",
    },
    {
        "text": "Acute pancreatitis secondary to gallstones. Managed with IV fluids and analgesia; lipase markedly elevated.",
        "entities": ["pancreatitis", "gallstones", "lipase"],
        "difficulty": "medium",
    },
    {
        "text": "Patient with multiple sclerosis relapse. Treated with high-dose intravenous methylprednisolone; MRI shows new lesions.",
        "entities": ["multiple sclerosis", "methylprednisolone", "magnetic resonance imaging"],
        "difficulty": "medium",
    },
    {
        "text": "Diabetic foot ulcer with surrounding cellulitis. Wound debridement performed; started on clindamycin.",
        "entities": ["foot ulcer", "cellulitis", "debridement", "clindamycin"],
        "difficulty": "medium",
    },
    {
        "text": "Patient with peptic ulcer disease and H. pylori infection. Started triple therapy: omeprazole, clarithromycin, and amoxicillin.",
        "entities": [
            "peptic ulcer disease",
            "Helicobacter pylori",
            "omeprazole",
            "clarithromycin",
            "amoxicillin",
        ],
        "difficulty": "medium",
    },
    {
        "text": "Patient presents with anaphylaxis after a bee sting. Treated with intramuscular epinephrine, antihistamines, and corticosteroids.",
        "entities": ["anaphylaxis", "epinephrine", "antihistamine", "corticosteroid"],
        "difficulty": "medium",
    },
    {
        "text": "Suspected pulmonary embolism confirmed on CT pulmonary angiography. Anticoagulation with heparin infusion initiated.",
        "entities": ["pulmonary embolism", "computed tomography", "heparin"],
        "difficulty": "medium",
    },
    {
        "text": "Patient with schizophrenia, poorly adherent. Restarted risperidone; offered long-acting injectable formulation.",
        "entities": ["schizophrenia", "risperidone"],
        "difficulty": "medium",
    },
    {
        "text": "Open reduction and internal fixation performed for a displaced femoral neck fracture. DVT prophylaxis with enoxaparin started.",
        "entities": [
            "femoral neck fracture",
            "open reduction and internal fixation",
            "deep vein thrombosis",
            "enoxaparin",
        ],
        "difficulty": "medium",
    },
    {
        "text": "Patient with ulcerative colitis in remission on mesalamine. Presents with a flare; prednisone course started.",
        "entities": ["ulcerative colitis", "mesalamine", "prednisone"],
        "difficulty": "medium",
    },
    # ----------------------------------------------------------------- hard
    {
        "text": "Pt c/o CP, h/o MI 2y ago. PMH: HTN, DM2, HLD. Meds: ASA, metoprolol, lisinopril, atorvastatin.",
        "entities": [
            "chest pain",
            "myocardial infarction",
            "hypertension",
            "diabetes",
            "hyperlipidemia",
            "aspirin",
            "metoprolol",
            "lisinopril",
            "atorvastatin",
        ],
        "difficulty": "hard",
    },
    {
        "text": "CXR shows R lower lobe infiltrate. Dx: CAP. Started on Levaquin 750mg qd x 5d. F/u in 1wk.",
        "entities": ["chest x-ray", "infiltrate", "community-acquired pneumonia", "levofloxacin"],
        "difficulty": "hard",
    },
    {
        "text": "Pt c/o SOB, orthopnea, PND. Hx CHF. Exam: JVD, bibasilar crackles, LE edema. Tx: IV Lasix, started ACEi.",
        "entities": [
            "shortness of breath",
            "orthopnea",
            "paroxysmal nocturnal dyspnea",
            "congestive heart failure",
            "jugular venous distension",
            "edema",
            "furosemide",
            "ACE inhibitor",
        ],
        "difficulty": "hard",
    },
    {
        "text": "63M w/ COPD exacerbation. Started DuoNeb, prednisone burst, and azithro. ABG: resp acidosis. Admit to floor.",
        "entities": [
            "chronic obstructive pulmonary disease",
            "ipratropium-albuterol",
            "prednisone",
            "azithromycin",
            "respiratory acidosis",
        ],
        "difficulty": "hard",
    },
    {
        "text": "Pt w/ DKA. Glu 480, AG 22, ketones +. Tx: IVF, insulin gtt, K repletion. Monitor q1h glucose.",
        "entities": [
            "diabetic ketoacidosis",
            "anion gap",
            "intravenous fluids",
            "insulin",
            "potassium",
        ],
        "difficulty": "hard",
    },
    {
        "text": "s/p CABG x3 POD2. On ASA, BB, statin. T-max 38.6, incision c/d/i. Cont incentive spirometry.",
        "entities": ["coronary artery bypass graft", "aspirin", "beta blocker", "statin"],
        "difficulty": "hard",
    },
    {
        "text": "Pt c/o RUQ pain, +Murphy's. US: cholelithiasis w/ GB wall thickening. Dx: acute cholecystitis. Surgery consulted for lap chole.",
        "entities": [
            "right upper quadrant pain",
            "ultrasound",
            "cholelithiasis",
            "cholecystitis",
            "cholecystectomy",
        ],
        "difficulty": "hard",
    },
    {
        "text": "AMS, fever, nuchal rigidity. LP: WBC 1200, low glucose, high protein. Dx: bacterial meningitis. Started ceftriaxone + vanc + dex.",
        "entities": [
            "altered mental status",
            "lumbar puncture",
            "bacterial meningitis",
            "ceftriaxone",
            "vancomycin",
            "dexamethasone",
        ],
        "difficulty": "hard",
    },
    {
        "text": "78F p/w fall, R hip pain, leg shortened/ER. XR: displaced femoral neck fx. Plan: ORIF vs hemiarthroplasty. DVT ppx w/ LMWH.",
        "entities": [
            "femoral neck fracture",
            "open reduction and internal fixation",
            "hemiarthroplasty",
            "deep vein thrombosis",
            "low molecular weight heparin",
        ],
        "difficulty": "hard",
    },
    {
        "text": "Pt w/ HTN urgency, BP 210/120, HA. No end-organ damage. Started PO labetalol, goal gradual reduction.",
        "entities": ["hypertensive urgency", "headache", "labetalol"],
        "difficulty": "hard",
    },
    {
        "text": "ESRD on HD, missed session. K 6.8, peaked T waves on EKG. Tx: Ca gluconate, insulin/D50, kayexalate; urgent HD.",
        "entities": [
            "end-stage renal disease",
            "hemodialysis",
            "hyperkalemia",
            "electrocardiogram",
            "calcium gluconate",
            "insulin",
            "sodium polystyrene sulfonate",
        ],
        "difficulty": "hard",
    },
    {
        "text": "Pt w/ afib RVR, rate 150s. Hemodynamically stable. Started dilt gtt, rate controlled. CHADS2 3, started on AC.",
        "entities": [
            "atrial fibrillation",
            "rapid ventricular response",
            "diltiazem",
            "anticoagulation",
        ],
        "difficulty": "hard",
    },
    {
        "text": "55M EtOH abuse p/w hematemesis. EGD: bleeding esophageal varices, banded. Started octreotide gtt and ceftriaxone ppx.",
        "entities": [
            "alcohol abuse",
            "hematemesis",
            "esophagogastroduodenoscopy",
            "esophageal varices",
            "octreotide",
            "ceftriaxone",
        ],
        "difficulty": "hard",
    },
    {
        "text": "Pt w/ PE on CTA, RV strain. Hemodynamically stable. Started heparin gtt, admit tele. Trend troponin/BNP.",
        "entities": [
            "pulmonary embolism",
            "computed tomography angiography",
            "heparin",
            "troponin",
        ],
        "difficulty": "hard",
    },
    {
        "text": "Septic shock 2/2 urosepsis. Lactate 4.2. Tx: 30cc/kg IVF, broad-spectrum abx (cefepime), norepi for MAP>65.",
        "entities": [
            "septic shock",
            "urosepsis",
            "lactate",
            "intravenous fluids",
            "cefepime",
            "norepinephrine",
        ],
        "difficulty": "hard",
    },
    {
        "text": "Pt w/ new AKI, Cr 3.1 from 0.9. FENa <1%, prerenal 2/2 dehydration. IVF bolus, hold nephrotoxics/NSAIDs.",
        "entities": [
            "acute kidney injury",
            "creatinine",
            "fractional excretion of sodium",
            "intravenous fluids",
        ],
        "difficulty": "hard",
    },
    {
        "text": "Pt p/w status epilepticus. Given IV lorazepam x2, loaded with levetiracetam. EEG ordered, intubated for airway.",
        "entities": [
            "status epilepticus",
            "lorazepam",
            "levetiracetam",
            "electroencephalogram",
            "intubation",
        ],
        "difficulty": "hard",
    },
    {
        "text": "Pt w/ cellulitis L LE, erythema spreading despite PO abx. Switched to IV vancomycin, marked borders, elevate limb.",
        "entities": ["cellulitis", "erythema", "vancomycin"],
        "difficulty": "hard",
    },
    {
        "text": "Pt s/p TKA POD1, w/ acute SOB and tachycardia. CTA r/o PE neg. CXR clear. Likely atelectasis; IS encouraged.",
        "entities": [
            "total knee arthroplasty",
            "shortness of breath",
            "tachycardia",
            "computed tomography angiography",
            "pulmonary embolism",
            "atelectasis",
        ],
        "difficulty": "hard",
    },
    {
        "text": "Pt w/ acute pancreatitis, lipase 1200, 2/2 gallstones. NPO, aggressive IVF, pain control w/ hydromorphone.",
        "entities": ["pancreatitis", "lipase", "gallstones", "hydromorphone"],
        "difficulty": "hard",
    },
    {
        "text": "DM2 w/ foot ulcer, probe to bone +. MRI: osteomyelitis. Started IV vanc + zosyn, vascular and podiatry consulted.",
        "entities": [
            "diabetes",
            "foot ulcer",
            "magnetic resonance imaging",
            "osteomyelitis",
            "vancomycin",
            "piperacillin-tazobactam",
        ],
        "difficulty": "hard",
    },
    {
        "text": "Pt w/ NSTEMI, trop 2.1 uptrending. Started DAPT (ASA+ticagrelor), heparin gtt, high-intensity statin. Cath in AM.",
        "entities": [
            "non-ST elevation myocardial infarction",
            "troponin",
            "dual antiplatelet therapy",
            "aspirin",
            "ticagrelor",
            "heparin",
            "statin",
        ],
        "difficulty": "hard",
    },
    {
        "text": "Pt c/o dysuria, frequency, suprapubic pain. UA: +LE, +nitrites. Dx: UTI. Started nitrofurantoin x5d.",
        "entities": ["dysuria", "urinary tract infection", "urinalysis", "nitrofurantoin"],
        "difficulty": "hard",
    },
    {
        "text": "Pt w/ hypothyroidism, TSH 14. Levothyroxine dose increased. Recheck TFTs in 6wks.",
        "entities": ["hypothyroidism", "thyroid stimulating hormone", "levothyroxine"],
        "difficulty": "hard",
    },
    {
        "text": "Pt w/ GIB, melena, Hgb 6.8. 2u PRBC transfused, started PPI gtt. EGD: bleeding duodenal ulcer, clipped.",
        "entities": [
            "gastrointestinal bleed",
            "melena",
            "packed red blood cells",
            "proton pump inhibitor",
            "esophagogastroduodenoscopy",
            "duodenal ulcer",
        ],
        "difficulty": "hard",
    },
    # --------------------------------------------------------------- expert
    {
        "text": "36F G2P1 at 38w c/ ROM x2h. FHR 140s, ctx q3min. Cx 4/80/-1. Plan: EFM, GBS+ so PCN, await SVD.",
        "entities": [
            "rupture of membranes",
            "fetal heart rate",
            "contractions",
            "cervix",
            "electronic fetal monitoring",
            "group B streptococcus",
            "penicillin",
        ],
        "difficulty": "expert",
    },
    {
        "text": "72M POD3 s/p Whipple. Now febrile, leukocytosis, RUQ drain bilious & up. Dx: bile leak. IR drain placed, abx broadened to mero.",
        "entities": [
            "pancreaticoduodenectomy",
            "leukocytosis",
            "bile leak",
            "interventional radiology",
            "meropenem",
        ],
        "difficulty": "expert",
    },
    {
        "text": "Pt w/ ARDS on AC/VC, P/F 95, FiO2 80%, PEEP 14. Proned, started cisatracurium, lung-protective Vt 6cc/kg. Trend ABG.",
        "entities": [
            "acute respiratory distress syndrome",
            "mechanical ventilation",
            "positive end-expiratory pressure",
            "cisatracurium",
        ],
        "difficulty": "expert",
    },
    {
        "text": "55F w/ SLE flare, nephritis on bx (class IV). Cr up, proteinuria 3.5g. Started pulse MTX-pred then MMF, hydroxychloroquine continued.",
        "entities": [
            "systemic lupus erythematosus",
            "lupus nephritis",
            "proteinuria",
            "methylprednisolone",
            "mycophenolate mofetil",
            "hydroxychloroquine",
        ],
        "difficulty": "expert",
    },
    {
        "text": "Pt w/ TTP: MAHA, thrombocytopenia, schistocytes, ADAMTS13 <10%. Started PLEX daily + high-dose steroids; hold plt transfusion.",
        "entities": [
            "thrombotic thrombocytopenic purpura",
            "microangiopathic hemolytic anemia",
            "thrombocytopenia",
            "plasma exchange",
            "corticosteroid",
        ],
        "difficulty": "expert",
    },
    {
        "text": "68M w/ cardiogenic shock 2/2 anterior STEMI. To cath, LAD 100% occ, stented. IABP placed, on dobutamine + norepi, intubated.",
        "entities": [
            "cardiogenic shock",
            "ST elevation myocardial infarction",
            "left anterior descending artery",
            "stent",
            "intra-aortic balloon pump",
            "dobutamine",
            "norepinephrine",
            "intubation",
        ],
        "difficulty": "expert",
    },
    {
        "text": "Pt p/w thyroid storm: AF, fever 40, agitation. Started PTU, then iodine (1h later), propranolol, hydrocortisone, cooling.",
        "entities": [
            "thyroid storm",
            "atrial fibrillation",
            "propylthiouracil",
            "iodine",
            "propranolol",
            "hydrocortisone",
        ],
        "difficulty": "expert",
    },
    {
        "text": "44M w/ massive PE, hypotensive, RV dilation on echo. Given systemic tPA, then heparin gtt. Sats improved, MAP recovered.",
        "entities": [
            "pulmonary embolism",
            "hypotension",
            "echocardiogram",
            "tissue plasminogen activator",
            "heparin",
        ],
        "difficulty": "expert",
    },
    {
        "text": "ICU pt w/ VAP, BAL +Pseudomonas. De-escalated to cefepime per sensitivities. CPIS improving, SBT passed, plan extubation.",
        "entities": [
            "ventilator-associated pneumonia",
            "bronchoalveolar lavage",
            "Pseudomonas",
            "cefepime",
            "spontaneous breathing trial",
            "extubation",
        ],
        "difficulty": "expert",
    },
    {
        "text": "Pt w/ tumor lysis post-induction for AML: K 6.5, PO4 high, Ca low, uric acid 12. Started rasburicase, aggressive IVF, monitor for AKI.",
        "entities": [
            "tumor lysis syndrome",
            "acute myeloid leukemia",
            "hyperkalemia",
            "hyperphosphatemia",
            "hypocalcemia",
            "rasburicase",
            "acute kidney injury",
        ],
        "difficulty": "expert",
    },
    {
        "text": "Neonate DOL1 w/ resp distress, grunting, retractions. CXR: ground-glass, low volumes. Dx: RDS. Surfactant given, on CPAP.",
        "entities": [
            "respiratory distress syndrome",
            "surfactant",
            "continuous positive airway pressure",
        ],
        "difficulty": "expert",
    },
    {
        "text": "Pt w/ NMS on antipsychotics: rigidity, hyperthermia, ↑CK, autonomic instability. D/C offending agent, started dantrolene + bromocriptine, IVF.",
        "entities": [
            "neuroleptic malignant syndrome",
            "creatine kinase",
            "dantrolene",
            "bromocriptine",
        ],
        "difficulty": "expert",
    },
    {
        "text": "Pt w/ DKA resolving but now AG closed, on SQ insulin transition w/ overlap. Cont K repletion, monitor for cerebral edema in peds.",
        "entities": [
            "diabetic ketoacidosis",
            "anion gap",
            "insulin",
            "potassium",
            "cerebral edema",
        ],
        "difficulty": "expert",
    },
    {
        "text": "62M w/ decompensated cirrhosis: SBP on tap (PMN>250), HE grade 2, AKI-HRS. Started cefotaxime + albumin, lactulose, octreotide/midodrine.",
        "entities": [
            "cirrhosis",
            "spontaneous bacterial peritonitis",
            "hepatic encephalopathy",
            "hepatorenal syndrome",
            "cefotaxime",
            "albumin",
            "lactulose",
            "octreotide",
            "midodrine",
        ],
        "difficulty": "expert",
    },
    {
        "text": "Pt w/ acute limb ischemia, 5 Ps. CTA: popliteal occlusion. Heparin bolus, taken emergently for thrombectomy/fasciotomy.",
        "entities": [
            "acute limb ischemia",
            "computed tomography angiography",
            "heparin",
            "thrombectomy",
            "fasciotomy",
        ],
        "difficulty": "expert",
    },
    {
        "text": "30F w/ severe preeclampsia at 34w: BP 170/110, proteinuria, HA, RUQ pain, plt 80. Started MgSO4, labetalol, betamethasone; plan delivery.",
        "entities": [
            "preeclampsia",
            "proteinuria",
            "thrombocytopenia",
            "magnesium sulfate",
            "labetalol",
            "betamethasone",
        ],
        "difficulty": "expert",
    },
    {
        "text": "Pt w/ refractory VT, multiple ICD shocks. Loaded amiodarone, then lidocaine. Sedated/intubated, electrolytes repleted, EP consulted for ablation.",
        "entities": [
            "ventricular tachycardia",
            "implantable cardioverter defibrillator",
            "amiodarone",
            "lidocaine",
            "ablation",
        ],
        "difficulty": "expert",
    },
    {
        "text": "Pt w/ DIC 2/2 sepsis: ↑PT/PTT, ↓fibrinogen, ↑D-dimer, schistocytes. Transfused FFP, cryo, platelets; treating underlying source w/ abx.",
        "entities": [
            "disseminated intravascular coagulation",
            "sepsis",
            "fibrinogen",
            "D-dimer",
            "fresh frozen plasma",
            "cryoprecipitate",
        ],
        "difficulty": "expert",
    },
    {
        "text": "Post-arrest pt ROSC after VF arrest. Initiated targeted temperature management 36C, started norepi, STEMI on EKG so emergent cath.",
        "entities": [
            "cardiac arrest",
            "ventricular fibrillation",
            "return of spontaneous circulation",
            "targeted temperature management",
            "norepinephrine",
            "ST elevation myocardial infarction",
        ],
        "difficulty": "expert",
    },
]
