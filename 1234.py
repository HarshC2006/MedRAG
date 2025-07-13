from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import networkx as nx
from collections import Counter
from med import build_knowledge, get_embedding

# --------------------- Load Knowledge Base ---------------------

result = build_knowledge()

triplets = result["triplets"]
G = result["graph"]
level2_clusters = result["level2_clusters"]
level1_clusters = result["level1_clusters"]
embeddings = result["embeddings"]

# --------------------- Load BioBERT ---------------------

model_name = "dmis-lab/biobert-base-cased-v1.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
biobert = AutoModel.from_pretrained(model_name)
biobert.eval()

# ------------------ Prepare Symptom Data ------------------

symptom_embeddings_dict = embeddings
symptom_nodes = list(embeddings.keys())

# ----------------- Find Similar Symptoms ------------------

def find_top_n_similar_symptoms(query: str, symptom_nodes: list, symptom_embeddings_dict: dict, n: int = 5, threshold: float = 0.5):
    query_embedding = get_embedding(query,tokenizer)

    # Compute similarities
    symptom_embeddings = []
    valid_symptom_names = []

    for symptom in symptom_nodes:
        if symptom in symptom_embeddings_dict:
            symptom_embeddings.append(symptom_embeddings_dict[symptom])
            valid_symptom_names.append(symptom)

    similarities = cosine_similarity([query_embedding], symptom_embeddings).flatten()

    top_n_symptoms = []
    for idx in similarities.argsort()[::-1]:
        name = valid_symptom_names[idx]
        sim = similarities[idx]
        if sim >= threshold:
            top_n_symptoms.append((name, float(sim)))
        if len(top_n_symptoms) == n:
            break

    return top_n_symptoms

# ------------------ Category Prediction ------------------

def find_closest_category_simple_upward(top_symptoms, categories, top_n):
    votes = Counter()
    
    for symptom in set(top_symptoms):
        if symptom not in G:
            continue
        
        for neighbor in G.successors(symptom):
            neighbor = neighbor.strip().replace(" ", "_").lower()
            if neighbor not in G:
                continue
            
            queue = [neighbor]
            visited = {neighbor}
            
            while queue:
                current = queue.pop(0)
                
                if current in categories:
                    votes[current] += 1
                    break
                
                for next_node in G.successors(current):
                    if next_node in visited:
                        continue
                    
                    edge_data = G.get_edge_data(current, next_node)
                    if edge_data:
                        for edge_attr in edge_data.values():
                            label = edge_attr.get('label', '')
                            if 'reverse' not in label.lower():
                                queue.append(next_node)
                                visited.add(next_node)
                                break
    
    return [cat for cat, _ in votes.most_common(top_n)]

query = "persistent cough with mucus and shortness of breath"
top_symptoms_with_scores = find_top_n_similar_symptoms(query, symptom_nodes, symptom_embeddings_dict, n=5)
top_symptom_names = [name for name, _ in top_symptoms_with_scores]
print("Top matching symptoms:", top_symptom_names)

# Step 2: Predict closest categories (level 1)
categories = [info["name"] for info in level1_clusters.values()]
predicted_categories = find_closest_category_simple_upward(top_symptom_names, categories, top_n=2)

print("Predicted categories:", predicted_categories)

from collections import defaultdict
import networkx as nx

def extract_from_level1_categories(graph: nx.MultiDiGraph, level1_categories: list) -> dict:
    """
    Given a list of Level 1 category names, extract:
    Level 1 → Level 2 → Level 3 hierarchy by traversing only downward 'is_a' edges.
    """
    hierarchy = {}

    for lvl1 in level1_categories:
        level2 = set()
        level3 = set()
        mapping = defaultdict(list)

        for lvl2 in graph.successors(lvl1):
            for attr in graph.get_edge_data(lvl1, lvl2).values():
                if attr.get("label") == "is_a":
                    level2.add(lvl2)

                    for disease in graph.successors(lvl2):
                        for inner_attr in graph.get_edge_data(lvl2, disease).values():
                            if inner_attr.get("label") == "is_a":
                                level3.add(disease)
                                mapping[lvl2].append(disease)

        hierarchy[lvl1] = {
            "level2": list(level2),
            "level3": list(level3),
            "mapping": dict(mapping)
        }

    return hierarchy

def generate_additional_info(hierarchy: dict, triplets: dict) -> list:
    additional_info = []
    for level_key in range(1,3):
        diseases = hierarchy.get(f"lvl{level_key}", {}).get("level3", [])

        for triplet in triplets:
            subject = triplet.get("subject", "")
            relation = triplet.get("relation", "")
            obj = triplet.get("object", {})
            obj_text = obj.get("fact")

            if subject in diseases:
                sentence = f"{subject} {relation} {obj_text}"
                additional_info.append(sentence)

    return additional_info

def get_system_prompt_for_RAGKG():
    return '''
        You are a knowledgeable medical assistant with expertise in pain management.
        Your tasks are:
        1. Analyse and refer to the retrieved similar patients' cases and knowledge graph which may be relevant to the diagnosis and assist with new patient cases.
2. Output of "Diagnoses" must come from : acute copd exacerbation infection, bronchiectasis, bronchiolitis, bronchitis, bronchospasm acute asthma exacerbation, pulmonary embolism, pulmonary neoplasm, spontaneous pneumothorax, urti, viral pharyngitis, whooping cough, acute laryngitis, acute pulmonary edema, croup, larygospasm, epiglottitis, pneumonia, atrial fibrillation, myocarditis, pericarditis, psvt, possible nstemi stemi, stable angina, unstable angina, gerd, boerhaave syndrome, pancreatic neoplasm, scombroid food poisoning, inguinal hernia, tuberculosis, hiv initial infection, ebola, influenza, chagas, acute otitis media, acute rhinosinusitis, allergic sinusitis, chronic rhinosinusitis, myasthenia gravis, guillain barre syndrome, cluster headache, acute dystonic reactions, sle, sarcoidosis, anaphylaxis, panic attack, spontaneous rib fracture, anemia.        3. You are given differences of diagnoses of similar symptoms or pain locations. Read that information as a reference to your diagnostic if applicable.
        4. Do mind the nuance between these factors of similar diagnosis with knowledge graph information and consider it when diagnose new patient's informtation.
        5. Ensure that the recommendations are evidence-based and consider the most recent and effective practices in pain management.
        6. The output should include four specific treatment-related fields:
           - "Diagnoses (related to pain)"
           - Explanations of diagnose
           - "Pain/General Physiotherapist Treatments\nSession No.: General Overview\n- Specific interventions/treatments"
           - "Pain Psychologist Treatments"
           - "Pain Medicine Treatments"
        7. In "Diagnoses", only output the diagnosis itself. Place all other explanations and analyses (if any) into "Explanations of diagnose".
        8. You can leave Psychologist Treatments blank if not applicable for the case, leaving text "Not applicable"
        9.If you think information is needed, guide the doctor to ask further questions which following areas to distinguish between the most likely diseases: Pain restriction; Location; Symptom. Seperate answers with ",". The output should only include aspects.
        10. The output should follow this structured format:
        

    ### Diagnoses
    1. **Diagnosis**: Answer.
    2. **Explanations of diagnose**: Answer.
    
    ### Instructive question
    1. **Questions**: Answer.
    
    ### Pain/General Physiotherapist Treatments
    1. **Session No.: General Overview**
        - **Specific interventions/treatments**:
        - **Goals**:
        - **Exercises**:
        - **Manual Therapy**:
        - **Techniques**:

    2. **Exercise Recommendations from the Exercise List**:

    ### Pain Psychologist Treatments(if applicable)
    1. **Treatment 1**: 
    
    ### Pain Medicine Treatments


    ### Recommendations for Further Evaluations
    1. **Evaluation 1**:
    '''

from llama_index.llms.groq import Groq

# Instantiate Groq LLM once outside the function (if not already)
llm = Groq(
    model="llama3-70b-8192",  # or "mixtral-8x7b-32768", etc.
    api_key="Groq_api_key"
)

system_prompt_RAGKG = get_system_prompt_for_RAGKG()
hierarchy=extract_from_level1_categories(G,predicted_categories)
additional_info = generate_additional_info(hierarchy,triplets)

prompt = f"""Query: {query}
Information from knowledge graph about relevant diagnoses, if you think the patient's disease is relevant from the suggestions provided by the atlas please refer to those details to distinguish similar diagnoses: {additional_info}
Now complete the tasks in that format."""

full_prompt = f"{system_prompt_RAGKG}\n\n{prompt}"
response = llm.complete(full_prompt)

print(response.text.strip())
