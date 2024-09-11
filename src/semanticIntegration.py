from rdflib import Graph, RDF, OWL, RDFS
import os
import numpy as np

def get_breast_cancer_knowledge():
    g = Graph()
    ontology_path = os.path.join(os.path.dirname(__file__), "BCGO.owl")
    g.parse(ontology_path, format="xml")
    
    print("Ontology loaded successfully")
    print("Number of triples in the ontology:", len(g))

    # Estrazione di classi e proprietà dall'ontologia
    classes = list(g.subjects(RDF.type, OWL.Class))
    properties = list(g.subjects(RDF.type, OWL.ObjectProperty))

    knowledge = {
        "ontology": g,
        "classes": classes,
        "properties": properties
    }

    print(f"Number of classes in the ontology: {len(classes)}")
    print(f"Number of properties in the ontology: {len(properties)}")

    return knowledge

def calculate_class_depth(g, class_uri):
    depth = 0
    current_class = class_uri
    while current_class != OWL.Thing:
        parent = g.value(subject=current_class, predicate=RDFS.subClassOf)
        if parent is None:
            break
        depth += 1
        current_class = parent
    return depth

def enrich_data_with_knowledge(X, knowledge):
    ontology = knowledge["ontology"]
    classes = knowledge["classes"]
    properties = knowledge["properties"]

    # Calcola la profondità massima delle classi
    max_depth = max(calculate_class_depth(ontology, c) for c in classes)

    # Funzione per calcolare il punteggio di rilevanza basato sulla profondità
    def relevance_score(depth):
        return 1 - (depth / max_depth) if max_depth > 0 else 1

    # Definizione dei fattori di rischio con punteggi più sfumati
    X["semantic_risk_factor_size"] = X["radius_mean"].apply(lambda x: (x - 15) / 5 if x > 15 else 0)
    X["semantic_risk_factor_area"] = X["area_mean"].apply(lambda x: (x - 500) / 100 if x > 500 else 0)
    X["semantic_risk_factor_texture"] = X["texture_mean"].apply(lambda x: (x - 20) / 5 if x > 20 else 0)

    # Cerchiamo informazioni specifiche nell'ontologia
    tubular_formation_prop = next((p for p in properties if "hasTubularFormationFromAllNeoplasm" in str(p)), None)
    nottingham_scoring_prop = next((p for p in properties if "hasNottinghamScoring" in str(p)), None)

    # Usiamo queste informazioni per creare nuove feature con punteggi di rilevanza
    if tubular_formation_prop:
        depth = calculate_class_depth(ontology, tubular_formation_prop)
        X["semantic_tubular_formation_relevance"] = relevance_score(depth)
    
    if nottingham_scoring_prop:
        depth = calculate_class_depth(ontology, nottingham_scoring_prop)
        X["semantic_nottingham_scoring_relevance"] = relevance_score(depth)

    # Cerchiamo classi rilevanti nell'ontologia
    breast_cancer_class = next((c for c in classes if "BreastCancer" in str(c)), None)
    tumor_class = next((c for c in classes if "Tumor" in str(c)), None)

    # Aggiungiamo queste informazioni come nuove feature con punteggi di rilevanza
    if breast_cancer_class:
        depth = calculate_class_depth(ontology, breast_cancer_class)
        X["semantic_breast_cancer_relevance"] = relevance_score(depth)
    
    if tumor_class:
        depth = calculate_class_depth(ontology, tumor_class)
        X["semantic_tumor_relevance"] = relevance_score(depth)

    # Combiniamo tutti i fattori di rischio usando una media ponderata
    risk_factors = [
        "semantic_risk_factor_size", 
        "semantic_risk_factor_area", 
        "semantic_risk_factor_texture"
    ]
    
    # Aggiungiamo solo le colonne che sono state effettivamente create
    if "semantic_tubular_formation_relevance" in X.columns:
        risk_factors.append("semantic_tubular_formation_relevance")
    if "semantic_nottingham_scoring_relevance" in X.columns:
        risk_factors.append("semantic_nottingham_scoring_relevance")
    if "semantic_breast_cancer_relevance" in X.columns:
        risk_factors.append("semantic_breast_cancer_relevance")
    if "semantic_tumor_relevance" in X.columns:
        risk_factors.append("semantic_tumor_relevance")

    X["semantic_combined_risk_factor"] = X[risk_factors].mean(axis=1)

    # Rimuoviamo le colonne costanti
    constant_columns = X.columns[X.nunique() == 1]
    X = X.drop(columns=constant_columns)

    print("Colonne semantiche aggiunte:", [col for col in X.columns if col.startswith("semantic_")])

    return X