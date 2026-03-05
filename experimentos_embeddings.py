# -----------------------------
# EXPERIMENTO INTRA/INTER Y PCA/t-SNE CON JSON EXTERNO
# -----------------------------

def load_person_embeddings_from_face_data(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    person_embeddings = {}
    # Si es lista de dicts con 'name' y 'embedding'
    if isinstance(data, list):
        for entry in data:
            name = entry.get("name")
            embedding = entry.get("embedding")
            if name and embedding:
                if name not in person_embeddings:
                    person_embeddings[name] = []
                person_embeddings[name].append(np.array(embedding))
    elif isinstance(data, dict):
        # Soporta el formato anterior
        for person, info in data.items():
            if isinstance(info, dict):
                if "embeddings" in info:
                    person_embeddings[person] = [np.array(e) for e in info["embeddings"]]
                elif "images" in info:
                    person_embeddings[person] = [np.array(img["embedding"]) for img in info["images"] if "embedding" in img]
            elif isinstance(info, list):
                person_embeddings[person] = [np.array(e) for e in info]
    return person_embeddings

# -----------------------------
# MAIN PARA FACE_DATA.JSON
# -----------------------------
def main_face_data_experiments(json_path):
    embeddings = load_person_embeddings_from_face_data(json_path)
    intra, inter = compute_distances(embeddings)
    plot_intra_inter(intra, inter)
    # Visualización PCA/t-SNE
    X = []
    labels = []
    for person, embs in embeddings.items():
        X.extend(embs)
        labels.extend([person] * len(embs))
    X = np.array(X)
    if len(X) == 0:
        print("No hay embeddings para visualizar.")
        return
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    print(f"Varianza explicada PCA: {pca.explained_variance_ratio_}")
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X)
    for data, name in [(X_pca, "pca"), (X_tsne, "tsne")]:
        plt.figure(figsize=(8, 6))
        for person in set(labels):
            idx = [i for i, l in enumerate(labels) if l == person]
            plt.scatter(data[idx, 0], data[idx, 1], label=person, s=80, alpha=0.8, edgecolors='k', linewidths=0.5)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.title(name.upper())
        plt.xlabel("Componente 1")
        plt.ylabel("Componente 2")
        plt.xlim(data[:, 0].min() - 1, data[:, 0].max() + 1)
        plt.ylim(data[:, 1].min() - 1, data[:, 1].max() + 1)
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        plt.savefig(f"{OUTPUT_DIR}/{name}_face_data.png", bbox_inches='tight')
        plt.close()
import os
from collections import defaultdict
import re
import cv2
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json

from deepface import DeepFace
from scipy.spatial.distance import cosine
from sklearn.metrics import roc_curve, auc
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from tqdm import tqdm

# -----------------------------
# CONFIGURACIÓN
# -----------------------------

DATASET_PATH = "FacesDataSet"
MODEL_NAME = "Facenet"
REPEAT_STABILITY = 10
OUTPUT_DIR = "resultados"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# CARGAR DATASET
# -----------------------------

def load_dataset(path):

    dataset = defaultdict(lambda: {"reference": None, "tests": []})
    # Recorrer subcarpetas (una por persona)
    for person_folder in os.listdir(path):
        person_path = os.path.join(path, person_folder)
        if not os.path.isdir(person_path):
            continue
        images = [f for f in os.listdir(person_path) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
        images.sort()  # Opcional: para que la referencia sea siempre la primera
        for idx, filename in enumerate(images):
            image_path = os.path.join(person_path, filename)
            if idx == 0:
                dataset[person_folder]["reference"] = image_path
            else:
                dataset[person_folder]["tests"].append(image_path)
        print(person_folder, len(dataset[person_folder]["tests"]))
    return dataset


# -----------------------------
# EXTRAER EMBEDDINGS
# -----------------------------

def extract_embeddings(dataset):
    embeddings = {}
    for person, imgs in dataset.items():
        person_embeddings = []
        # Procesar referencia si existe
        if imgs["reference"]:
            emb = DeepFace.represent(
                imgs["reference"],
                model_name=MODEL_NAME,
                enforce_detection=False
            )[0]["embedding"]
            person_embeddings.append(np.array(emb))
        # Procesar tests
        for img_path in imgs["tests"]:
            emb = DeepFace.represent(
                img_path,
                model_name=MODEL_NAME,
                enforce_detection=False
            )[0]["embedding"]
            person_embeddings.append(np.array(emb))
        embeddings[person] = np.array(person_embeddings)
    return embeddings


# -----------------------------
# EXPERIMENTO 1: ESTABILIDAD
# -----------------------------

def experiment_stability(image_path):
    # Solo calcular embeddings si el archivo no existe
    json_path = os.path.join(OUTPUT_DIR, "experimento1.json")
    if os.path.exists(json_path):
        print("El archivo experimento1.json ya existe. No se recalculan los embeddings.")
        return

    embeddings_dict = {}
    dist_stats = {}
    all_distances = []
    image_paths = []

    dataset = load_dataset(DATASET_PATH)
    for person, imgs in dataset.items():
        if imgs["reference"]:
            image_paths.append(imgs["reference"])
        image_paths.extend(imgs["tests"])

    for idx, img_path in enumerate(image_paths):
        emb_list = []
        for _ in range(REPEAT_STABILITY):
            emb = DeepFace.represent(
                img_path,
                model_name=MODEL_NAME,
                enforce_detection=False
            )[0]["embedding"]
            emb_list.append(np.array(emb))
        embeddings_dict[img_path] = [emb.tolist() for emb in emb_list]
        distances = []
        if len(emb_list) > 1:
            for i in range(len(emb_list)):
                for j in range(i + 1, len(emb_list)):
                    distances.append(cosine(emb_list[i], emb_list[j]))
        if distances:
            dist_stats[img_path] = {
                "media": float(np.mean(distances)),
                "std": float(np.std(distances)),
                "min": float(np.min(distances)),
                "max": float(np.max(distances)),
                "distancias": distances
            }
            all_distances.extend(distances)
        else:
            dist_stats[img_path] = {
                "media": None,
                "std": None,
                "min": None,
                "max": None,
                "distancias": []
            }

        if distances:
            plt.figure()
            sns.histplot(distances, bins=20)
            plt.title(f"Estabilidad embedding: {os.path.basename(img_path)}")
            plt.xlabel("Distancia coseno")
            plt.savefig(f"{OUTPUT_DIR}/estabilidad_{idx}_hist.png")
            plt.close()

            plt.figure()
            sns.boxplot(x=distances)
            plt.title(f"Boxplot embedding: {os.path.basename(img_path)}")
            plt.xlabel("Distancia coseno")
            plt.savefig(f"{OUTPUT_DIR}/estabilidad_{idx}_box.png")
            plt.close()

        print(f"Imagen: {img_path}")
        if distances:
            print(f"Media: {dist_stats[img_path]['media']:.6f}")
            print(f"Desviación estándar: {dist_stats[img_path]['std']:.6f}")
            print(f"Media ± std: {dist_stats[img_path]['media']:.6f} ± {dist_stats[img_path]['std']:.6f}")
        else:
            print("No hay suficientes embeddings para calcular distancias.")

    if all_distances:
        plt.figure()
        sns.histplot(all_distances, bins=20)
        plt.title("Estabilidad global del embedding")
        plt.xlabel("Distancia coseno")
        plt.savefig(f"{OUTPUT_DIR}/estabilidad_global_hist.png")
        plt.close()

        plt.figure()
        sns.boxplot(x=all_distances)
        plt.title("Boxplot global del embedding")
        plt.xlabel("Distancia coseno")
        plt.savefig(f"{OUTPUT_DIR}/estabilidad_global_box.png")
        plt.close()

        print("Estabilidad global media:", np.mean(all_distances))
        print("Estabilidad global std:", np.std(all_distances))
        print(f"Media ± std: {np.mean(all_distances):.6f} ± {np.std(all_distances):.6f}")
    else:
        print("No hay distancias globales para mostrar.")

    output_json = {
        "embeddings": embeddings_dict,
        "stats": dist_stats,
        "global": {
            "media": float(np.mean(all_distances)) if all_distances else None,
            "std": float(np.std(all_distances)) if all_distances else None,
            "min": float(np.min(all_distances)) if all_distances else None,
            "max": float(np.max(all_distances)) if all_distances else None,
            "distancias": all_distances
        }
    }
    with open(json_path, "w") as f:
        json.dump(output_json, f, indent=2)


# -----------------------------
# EXPERIMENTO 2-3: INTRA / INTER
# -----------------------------

def load_embeddings_from_json(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    # embeddings_dict: {img_path: [embedding1, embedding2, ...]}
    embeddings_dict = data["embeddings"]
    # Agrupar por persona
    person_embeddings = {}
    for img_path, emb_lists in embeddings_dict.items():
        # Extraer nombre de persona desde el path
        basename = os.path.basename(img_path)
        match = re.match(r"(.+?)(\d+)", os.path.splitext(basename)[0])
        if match:
            person = match.group(1)
            # Usar solo el primer embedding (para intra/inter)
            if person not in person_embeddings:
                person_embeddings[person] = []
            # Si emb_lists es una lista de embeddings (por estabilidad), tomar el primero
            if isinstance(emb_lists[0], list):
                person_embeddings[person].append(np.array(emb_lists[0]))
            else:
                person_embeddings[person].append(np.array(emb_lists))
    return person_embeddings

def compute_distances(embeddings):
    intra = []
    inter = []
    persons = list(embeddings.keys())
    # intra
    for person in persons:
        embs = embeddings[person]
        for i in range(len(embs)):
            for j in range(i + 1, len(embs)):
                intra.append(cosine(embs[i], embs[j]))
    # inter
    for i in range(len(persons)):
        for j in range(i + 1, len(persons)):
            for emb1 in embeddings[persons[i]]:
                for emb2 in embeddings[persons[j]]:
                    inter.append(cosine(emb1, emb2))
    return intra, inter

def plot_intra_inter(intra, inter):
    plt.figure()
    sns.histplot(intra, color="blue", label="Intra", stat="density", bins=30)
    sns.histplot(inter, color="red", label="Inter", stat="density", bins=30)
    plt.legend()
    plt.title("Distancias intra vs inter")
    plt.savefig(f"{OUTPUT_DIR}/intra_inter.png")
    plt.close()

    # Scatter plot intra/inter
    plt.figure()
    plt.scatter(["intra"]*len(intra), intra, color="blue", alpha=0.5, label="Intra")
    plt.scatter(["inter"]*len(inter), inter, color="red", alpha=0.5, label="Inter")
    plt.ylabel("Distancia coseno")
    plt.title("Scatter Intra vs Inter")
    plt.legend()
    plt.savefig(f"{OUTPUT_DIR}/intra_inter_scatter.png")
    plt.close()

    # PCA/t-SNE de todas las embeddings
    # No tiene sentido hacer PCA/t-SNE sobre distancias, sino sobre embeddings
    # Por eso, la visualización se hace en visualize_embeddings


# -----------------------------
# EXPERIMENTO 4: ROC
# -----------------------------

def compute_roc_from_json(json_path):
    # Cargar embeddings y calcular distancias intra/inter
    embeddings = load_embeddings_from_json(json_path)
    intra, inter = compute_distances(embeddings)
    y_true = [1] * len(intra) + [0] * len(inter)
    scores = intra + inter

    fpr, tpr, _ = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], "--")

    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("Curva ROC")
    plt.legend()

    plt.savefig(f"{OUTPUT_DIR}/roc.png")
    plt.close()

    print("AUC:", roc_auc)


# -----------------------------
# EXPERIMENTO 5: PCA / t-SNE
# -----------------------------

def visualize_embeddings_from_json(json_path):
    embeddings = load_embeddings_from_json(json_path)
    X = []
    labels = []
    for person, embs in embeddings.items():
        X.extend(embs)
        labels.extend([person] * len(embs))
    X = np.array(X)
    if len(X) == 0:
        print("No hay embeddings para visualizar.")
        return
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X)
    for data, name in [(X_pca, "pca"), (X_tsne, "tsne")]:
        plt.figure(figsize=(8, 6))
        for person in set(labels):
            idx = [i for i, l in enumerate(labels) if l == person]
            plt.scatter(data[idx, 0], data[idx, 1], label=person, s=80, alpha=0.8, edgecolors='k', linewidths=0.5)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.title(name.upper())
        plt.xlabel("Componente 1")
        plt.ylabel("Componente 2")
        plt.xlim(data[:, 0].min() - 1, data[:, 0].max() + 1)
        plt.ylim(data[:, 1].min() - 1, data[:, 1].max() + 1)
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        plt.savefig(f"{OUTPUT_DIR}/{name}.png", bbox_inches='tight')
        plt.close()


# -----------------------------
# MAIN
# -----------------------------

if __name__ == "__main__":

    # Para usar face_data.json, solo intra/inter y visualizaciones
    face_data_path = os.path.join(OUTPUT_DIR, "face_data.json")
    if os.path.exists(face_data_path):
        print("Usando face_data.json para experimentos intra/inter y PCA/t-SNE...")
        main_face_data_experiments(face_data_path)
        print("Experimentos con face_data.json completados.")
    else:
        # Flujo normal con experimento1.json
        json_path = os.path.join(OUTPUT_DIR, "experimento1.json")
        dataset = load_dataset(DATASET_PATH)
        first_person = next(iter(dataset))
        first_image = dataset[first_person]["reference"]
        experiment_stability(first_image)
        if os.path.exists(json_path):
            embeddings = load_embeddings_from_json(json_path)
            print("Embeddings cargados desde experimento1.json")
        else:
            print("No se encontró experimento1.json, extrayendo embeddings...")
            embeddings = extract_embeddings(dataset)
        intra, inter = compute_distances(embeddings)
        plot_intra_inter(intra, inter)
        compute_roc_from_json(json_path)
        visualize_embeddings_from_json(json_path)
        print("Experimentos completados.")
