import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from deepface import DeepFace
from scipy.spatial.distance import cosine
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
	accuracy_score,
	auc,
	confusion_matrix,
	precision_recall_fscore_support,
	roc_curve,
)
from tqdm import tqdm


# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------

DATASET_PATH = "FacesDataSet"
OUTPUT_DIR = "resultados"
MODEL_NAME = "Facenet"
DETECTOR_BACKEND = "opencv"

N_VALUES = [1, 3, 6, 12]
REPETITIONS = 30
BASE_SEED = 42

DISTANCE_METRIC = "cosine"  # options: cosine, euclidean
GALLERY_AGGREGATION = "mean"  # options: mean, min

EMBEDDINGS_CACHE_FILE = os.path.join(OUTPUT_DIR, "embeddings_facenet_cache.json")
METRICS_CSV_FILE = os.path.join(OUTPUT_DIR, "experimento2_metrics_por_repeticion.csv")
SUMMARY_JSON_FILE = os.path.join(OUTPUT_DIR, "experimento2_resumen.json")


# -----------------------------------------------------------------------------
# DATA CLASSES
# -----------------------------------------------------------------------------


@dataclass
class ImageEmbedding:
	person: str
	image_path: str
	embedding: np.ndarray


@dataclass
class SplitData:
	gallery: Dict[str, List[ImageEmbedding]]
	query: List[ImageEmbedding]


# -----------------------------------------------------------------------------
# UTILITIES
# -----------------------------------------------------------------------------


def ensure_output_dir() -> None:
	os.makedirs(OUTPUT_DIR, exist_ok=True)


def l2_normalize(vector: np.ndarray) -> np.ndarray:
	norm = np.linalg.norm(vector)
	if norm == 0.0:
		return vector
	return vector / norm


def compute_distance(v1: np.ndarray, v2: np.ndarray, metric: str = DISTANCE_METRIC) -> float:
	if metric == "cosine":
		return float(cosine(v1, v2))
	if metric == "euclidean":
		return float(np.linalg.norm(v1 - v2))
	raise ValueError(f"Unsupported distance metric: {metric}")


def list_images_by_person(dataset_path: str) -> Dict[str, List[str]]:
	people_images: Dict[str, List[str]] = {}

	if not os.path.isdir(dataset_path):
		raise FileNotFoundError(f"Dataset folder not found: {dataset_path}")

	for person in sorted(os.listdir(dataset_path)):
		person_path = os.path.join(dataset_path, person)
		if not os.path.isdir(person_path):
			continue

		image_files = [
			os.path.join(person_path, filename)
			for filename in sorted(os.listdir(person_path))
			if filename.lower().endswith((".jpg", ".jpeg", ".png"))
		]

		if image_files:
			people_images[person] = image_files

	return people_images


def validate_dataset(people_images: Dict[str, List[str]], required_per_person: int = 12) -> None:
	if len(people_images) == 0:
		raise ValueError("No people with images were found in dataset.")

	too_small = {person: len(images) for person, images in people_images.items() if len(images) < required_per_person}
	if too_small:
		details = ", ".join(f"{person}={count}" for person, count in sorted(too_small.items()))
		raise ValueError(
			"Some people do not have enough images for N up to 12. "
			f"Expected >= {required_per_person}, found: {details}"
		)


def extract_embedding(image_path: str) -> np.ndarray:
	# DeepFace handles detection + alignment before generating the embedding.
	result = DeepFace.represent(
		img_path=image_path,
		model_name=MODEL_NAME,
		detector_backend=DETECTOR_BACKEND,
		enforce_detection=False,
		align=True,
	)
	vector = np.array(result[0]["embedding"], dtype=np.float64)
	return l2_normalize(vector)


def build_or_load_embeddings(people_images: Dict[str, List[str]], cache_file: str) -> Dict[str, List[ImageEmbedding]]:
	if os.path.exists(cache_file):
		with open(cache_file, "r", encoding="utf-8") as f:
			cache = json.load(f)

		data = cache.get("data", {})
		parsed: Dict[str, List[ImageEmbedding]] = {}
		for person, items in data.items():
			parsed[person] = [
				ImageEmbedding(
					person=person,
					image_path=item["image_path"],
					embedding=np.array(item["embedding"], dtype=np.float64),
				)
				for item in items
			]
		return parsed

	parsed: Dict[str, List[ImageEmbedding]] = {}
	for person in tqdm(sorted(people_images.keys()), desc="Extrayendo embeddings"):
		parsed[person] = []
		for image_path in people_images[person]:
			emb = extract_embedding(image_path)
			parsed[person].append(ImageEmbedding(person=person, image_path=image_path, embedding=emb))

	serializable = {
		"metadata": {
			"created_at": datetime.utcnow().isoformat(),
			"model_name": MODEL_NAME,
			"detector_backend": DETECTOR_BACKEND,
			"distance_metric": DISTANCE_METRIC,
			"dataset_path": DATASET_PATH,
		},
		"data": {
			person: [
				{
					"image_path": item.image_path,
					"embedding": item.embedding.tolist(),
				}
				for item in items
			]
			for person, items in parsed.items()
		},
	}

	with open(cache_file, "w", encoding="utf-8") as f:
		json.dump(serializable, f, indent=2)

	return parsed


def split_gallery_query(
	person_embeddings: Dict[str, List[ImageEmbedding]],
	n_gallery: int,
	rng: np.random.Generator,
) -> SplitData:
	gallery: Dict[str, List[ImageEmbedding]] = {}
	query: List[ImageEmbedding] = []

	for person, items in person_embeddings.items():
		if len(items) < n_gallery + 1:
			raise ValueError(
				f"Person '{person}' does not have enough images for n_gallery={n_gallery}. "
				f"Found {len(items)}"
			)

		indices = np.arange(len(items))
		gallery_indices = set(rng.choice(indices, size=n_gallery, replace=False).tolist())

		gallery[person] = [items[idx] for idx in sorted(gallery_indices)]
		for idx, item in enumerate(items):
			if idx not in gallery_indices:
				query.append(item)

	return SplitData(gallery=gallery, query=query)


def predict_identity(
	query_emb: np.ndarray,
	gallery: Dict[str, List[ImageEmbedding]],
	distance_metric: str,
	aggregation: str,
) -> str:
	best_person = None
	best_score = np.inf

	for person, refs in gallery.items():
		distances = [compute_distance(query_emb, ref.embedding, metric=distance_metric) for ref in refs]
		if aggregation == "mean":
			score = float(np.mean(distances))
		elif aggregation == "min":
			score = float(np.min(distances))
		else:
			raise ValueError(f"Unsupported aggregation method: {aggregation}")

		if score < best_score:
			best_score = score
			best_person = person

	if best_person is None:
		raise RuntimeError("Prediction failed: no gallery references available.")
	return best_person


def compute_multiclass_metrics(
	y_true: Sequence[str],
	y_pred: Sequence[str],
	labels: Sequence[str],
) -> Dict[str, float]:
	acc = accuracy_score(y_true, y_pred)
	p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
		y_true, y_pred, average="macro", zero_division=0
	)
	p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
		y_true, y_pred, average="weighted", zero_division=0
	)

	cm = confusion_matrix(y_true, y_pred, labels=labels)

	return {
		"accuracy": float(acc),
		"precision_macro": float(p_macro),
		"recall_macro": float(r_macro),
		"f1_macro": float(f1_macro),
		"precision_weighted": float(p_weighted),
		"recall_weighted": float(r_weighted),
		"f1_weighted": float(f1_weighted),
		"confusion_matrix": cm,
	}


def centroid_distances_for_split(
	split: SplitData,
	metric: str,
) -> Tuple[List[float], List[float]]:
	centroids: Dict[str, np.ndarray] = {}
	for person, refs in split.gallery.items():
		centroid = np.mean([r.embedding for r in refs], axis=0)
		centroids[person] = l2_normalize(centroid)

	intra: List[float] = []
	inter: List[float] = []

	for item in split.query:
		same_centroid = centroids[item.person]
		intra.append(compute_distance(item.embedding, same_centroid, metric=metric))

		for person, centroid in centroids.items():
			if person == item.person:
				continue
			inter.append(compute_distance(item.embedding, centroid, metric=metric))

	return intra, inter


def build_verification_scores(
	split: SplitData,
	metric: str,
	rng: np.random.Generator,
	negative_ratio: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
	labels: List[int] = []
	scores: List[float] = []

	people = sorted(split.gallery.keys())

	for item in split.query:
		positives = split.gallery[item.person]
		for ref in positives:
			distance = compute_distance(item.embedding, ref.embedding, metric=metric)
			score = 1.0 - distance
			labels.append(1)
			scores.append(score)

		negative_candidates: List[ImageEmbedding] = []
		for person in people:
			if person == item.person:
				continue
			negative_candidates.extend(split.gallery[person])

		if not negative_candidates:
			continue

		n_negatives = min(len(negative_candidates), max(1, len(positives) * negative_ratio))
		idx = rng.choice(len(negative_candidates), size=n_negatives, replace=False)
		for i in idx:
			ref = negative_candidates[int(i)]
			distance = compute_distance(item.embedding, ref.embedding, metric=metric)
			score = 1.0 - distance
			labels.append(0)
			scores.append(score)

	return np.array(labels), np.array(scores)


def compute_roc_stats(y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, object]:
	fpr, tpr, thresholds = roc_curve(y_true, y_score)
	roc_auc = auc(fpr, tpr)

	# Youden J maximizes TPR - FPR and yields the best operating point.
	youden_idx = int(np.argmax(tpr - fpr))
	best_score_threshold = float(thresholds[youden_idx])
	best_distance_threshold = float(1.0 - best_score_threshold)

	y_pred = (y_score >= best_score_threshold).astype(int)
	verif_acc = float(np.mean(y_pred == y_true))

	return {
		"fpr": fpr,
		"tpr": tpr,
		"thresholds": thresholds,
		"auc": float(roc_auc),
		"youden_index": float((tpr[youden_idx] - fpr[youden_idx])),
		"best_score_threshold": best_score_threshold,
		"best_distance_threshold": best_distance_threshold,
		"tpr_at_best": float(tpr[youden_idx]),
		"fpr_at_best": float(fpr[youden_idx]),
		"verification_accuracy_at_best": verif_acc,
	}


def aggregate_roc_curves(curves: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
	grid = np.linspace(0.0, 1.0, 200)
	interp_tprs = []
	aucs = []

	for curve in curves:
		interp = np.interp(grid, curve["fpr"], curve["tpr"])
		interp[0] = 0.0
		interp_tprs.append(interp)
		aucs.append(curve["auc"])

	tpr_mean = np.mean(np.array(interp_tprs), axis=0)
	tpr_std = np.std(np.array(interp_tprs), axis=0)

	return {
		"fpr_grid": grid,
		"tpr_mean": tpr_mean,
		"tpr_std": tpr_std,
		"auc_mean": float(np.mean(aucs)),
		"auc_std": float(np.std(aucs)),
	}


# -----------------------------------------------------------------------------
# VISUALIZATION
# -----------------------------------------------------------------------------


def save_confusion_heatmap(cm: np.ndarray, labels: Sequence[str], n_value: int) -> None:
	plt.figure(figsize=(10, 8))
	sns.heatmap(
		cm,
		cmap="Blues",
		xticklabels=labels,
		yticklabels=labels,
		annot=False,
		cbar=True,
	)
	plt.title(f"Matriz de confusion promedio - N={n_value}")
	plt.xlabel("Prediccion")
	plt.ylabel("Real")
	plt.tight_layout()
	plt.savefig(os.path.join(OUTPUT_DIR, f"experimento2_confusion_N{n_value}.png"), dpi=200)
	plt.close()


def save_accuracy_vs_n(summary_rows: List[Dict[str, float]]) -> None:
	df = pd.DataFrame(summary_rows).sort_values("N")
	plt.figure(figsize=(8, 5))
	plt.errorbar(df["N"], df["accuracy_mean"], yerr=df["accuracy_std"], marker="o", capsize=4)
	plt.title("Accuracy vs N (imagenes por persona en galeria)")
	plt.xlabel("N")
	plt.ylabel("Accuracy")
	plt.ylim(0, 1)
	plt.grid(alpha=0.3)
	plt.tight_layout()
	plt.savefig(os.path.join(OUTPUT_DIR, "experimento2_accuracy_vs_n.png"), dpi=200)
	plt.close()


def save_metrics_boxplots(metrics_df: pd.DataFrame) -> None:
	metrics_to_plot = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
	for metric_name in metrics_to_plot:
		plt.figure(figsize=(8, 5))
		sns.boxplot(data=metrics_df, x="N", y=metric_name)
		sns.stripplot(data=metrics_df, x="N", y=metric_name, color="black", alpha=0.25, size=3)
		plt.title(f"{metric_name} por N")
		plt.ylim(0, 1)
		plt.tight_layout()
		plt.savefig(os.path.join(OUTPUT_DIR, f"experimento2_boxplot_{metric_name}.png"), dpi=200)
		plt.close()


def save_roc_plot(roc_aggregated_by_n: Dict[int, Dict[str, np.ndarray]]) -> None:
	plt.figure(figsize=(8, 6))
	for n_value in sorted(roc_aggregated_by_n.keys()):
		data = roc_aggregated_by_n[n_value]
		label = f"N={n_value} | AUC={data['auc_mean']:.3f} +/- {data['auc_std']:.3f}"
		plt.plot(data["fpr_grid"], data["tpr_mean"], label=label)
	plt.plot([0, 1], [0, 1], "k--", alpha=0.7)
	plt.xlabel("FPR")
	plt.ylabel("TPR")
	plt.title("Curvas ROC promedio por N")
	plt.legend(loc="lower right")
	plt.grid(alpha=0.3)
	plt.tight_layout()
	plt.savefig(os.path.join(OUTPUT_DIR, "experimento2_roc_por_n.png"), dpi=200)
	plt.close()


def save_intra_inter_distributions(separability_rows: List[Dict[str, float]]) -> None:
	sep_df = pd.DataFrame(separability_rows)

	plt.figure(figsize=(8, 5))
	sns.boxplot(data=sep_df, x="N", y="intra_mean")
	plt.title("Dispersion intra-clase por N")
	plt.ylabel("Distancia")
	plt.tight_layout()
	plt.savefig(os.path.join(OUTPUT_DIR, "experimento2_intra_por_n.png"), dpi=200)
	plt.close()

	plt.figure(figsize=(8, 5))
	sns.boxplot(data=sep_df, x="N", y="inter_mean")
	plt.title("Separacion inter-clase por N")
	plt.ylabel("Distancia")
	plt.tight_layout()
	plt.savefig(os.path.join(OUTPUT_DIR, "experimento2_inter_por_n.png"), dpi=200)
	plt.close()

	plt.figure(figsize=(8, 5))
	sns.boxplot(data=sep_df, x="N", y="margin_inter_minus_intra")
	plt.title("Margen (inter - intra) por N")
	plt.ylabel("Distancia")
	plt.tight_layout()
	plt.savefig(os.path.join(OUTPUT_DIR, "experimento2_margin_por_n.png"), dpi=200)
	plt.close()


def save_embedding_projection(person_embeddings: Dict[str, List[ImageEmbedding]], method: str = "pca") -> None:
	vectors = []
	labels = []
	for person, items in sorted(person_embeddings.items()):
		for item in items:
			vectors.append(item.embedding)
			labels.append(person)

	X = np.array(vectors)
	if len(X) == 0:
		return

	if method == "pca":
		reducer = PCA(n_components=2)
		coords = reducer.fit_transform(X)
	elif method == "tsne":
		reducer = TSNE(n_components=2, random_state=BASE_SEED)
		coords = reducer.fit_transform(X)
	else:
		raise ValueError("Projection method must be 'pca' or 'tsne'.")

	plt.figure(figsize=(10, 7))
	unique_labels = sorted(set(labels))
	for person in unique_labels:
		idx = [i for i, x in enumerate(labels) if x == person]
		plt.scatter(coords[idx, 0], coords[idx, 1], label=person, s=24, alpha=0.75)
	plt.title(f"Embeddings 2D ({method.upper()})")
	plt.xlabel("Comp 1")
	plt.ylabel("Comp 2")
	plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
	plt.tight_layout(rect=[0, 0, 0.84, 1])
	plt.savefig(os.path.join(OUTPUT_DIR, f"experimento2_{method}.png"), dpi=200)
	plt.close()


# -----------------------------------------------------------------------------
# MAIN EXPERIMENT
# -----------------------------------------------------------------------------


def run_experiment() -> None:
	ensure_output_dir()
	np.random.seed(BASE_SEED)

	people_images = list_images_by_person(DATASET_PATH)
	validate_dataset(people_images, required_per_person=max(N_VALUES))
	person_embeddings = build_or_load_embeddings(people_images, EMBEDDINGS_CACHE_FILE)

	labels = sorted(person_embeddings.keys())
	min_images_per_person = min(len(items) for items in person_embeddings.values())

	metrics_rows: List[Dict[str, float]] = []
	separability_rows: List[Dict[str, float]] = []
	summary_rows: List[Dict[str, float]] = []
	roc_aggregated_by_n: Dict[int, Dict[str, np.ndarray]] = {}
	summary_per_n: Dict[str, Dict[str, object]] = {}

	for n_value in N_VALUES:
		n_effective = min(n_value, min_images_per_person - 1)
		if n_effective < 1:
			raise ValueError(
				"Dataset must contain at least 2 images per person to build non-overlapping gallery/query splits."
			)

		if n_effective != n_value:
			print(
				f"\nProcesando N={n_value} (ajustado a N efectivo={n_effective} para evitar solapamiento)..."
			)
		else:
			print(f"\nProcesando N={n_value}...")
		cms: List[np.ndarray] = []
		roc_curves_for_n: List[Dict[str, np.ndarray]] = []
		thresholds_dist: List[float] = []
		tprs_best: List[float] = []
		fprs_best: List[float] = []
		verif_accs_best: List[float] = []

		for rep in tqdm(range(REPETITIONS), desc=f"N={n_value}"):
			rng = np.random.default_rng(BASE_SEED + (n_value * 1000) + rep)
			split = split_gallery_query(person_embeddings, n_gallery=n_effective, rng=rng)

			y_true: List[str] = []
			y_pred: List[str] = []

			for item in split.query:
				pred = predict_identity(
					query_emb=item.embedding,
					gallery=split.gallery,
					distance_metric=DISTANCE_METRIC,
					aggregation=GALLERY_AGGREGATION,
				)
				y_true.append(item.person)
				y_pred.append(pred)

			metric_data = compute_multiclass_metrics(y_true, y_pred, labels=labels)
			cm = metric_data.pop("confusion_matrix")
			cms.append(cm)

			metrics_rows.append(
				{
					"N": n_value,
					"repetition": rep,
					**metric_data,
				}
			)

			intra, inter = centroid_distances_for_split(split, metric=DISTANCE_METRIC)
			separability_rows.append(
				{
					"N": n_value,
					"repetition": rep,
					"intra_mean": float(np.mean(intra)) if intra else np.nan,
					"intra_std": float(np.std(intra)) if intra else np.nan,
					"inter_mean": float(np.mean(inter)) if inter else np.nan,
					"inter_std": float(np.std(inter)) if inter else np.nan,
					"margin_inter_minus_intra": float(np.mean(inter) - np.mean(intra)) if intra and inter else np.nan,
				}
			)

			y_verif_true, y_verif_score = build_verification_scores(
				split,
				metric=DISTANCE_METRIC,
				rng=rng,
				negative_ratio=1,
			)
			roc_stats = compute_roc_stats(y_verif_true, y_verif_score)
			roc_curves_for_n.append(
				{
					"fpr": roc_stats["fpr"],
					"tpr": roc_stats["tpr"],
					"auc": roc_stats["auc"],
				}
			)
			thresholds_dist.append(float(roc_stats["best_distance_threshold"]))
			tprs_best.append(float(roc_stats["tpr_at_best"]))
			fprs_best.append(float(roc_stats["fpr_at_best"]))
			verif_accs_best.append(float(roc_stats["verification_accuracy_at_best"]))

		cm_mean = np.mean(np.array(cms), axis=0)
		save_confusion_heatmap(cm_mean, labels=labels, n_value=n_value)

		roc_agg = aggregate_roc_curves(roc_curves_for_n)
		roc_aggregated_by_n[n_value] = roc_agg

		df_n = pd.DataFrame([row for row in metrics_rows if row["N"] == n_value])
		summary_row = {
			"N": n_value,
			"N_effective": n_effective,
			"accuracy_mean": float(df_n["accuracy"].mean()),
			"accuracy_std": float(df_n["accuracy"].std()),
			"precision_macro_mean": float(df_n["precision_macro"].mean()),
			"recall_macro_mean": float(df_n["recall_macro"].mean()),
			"f1_macro_mean": float(df_n["f1_macro"].mean()),
			"auc_mean": float(roc_agg["auc_mean"]),
			"auc_std": float(roc_agg["auc_std"]),
			"youden_distance_threshold_mean": float(np.mean(thresholds_dist)),
			"youden_distance_threshold_std": float(np.std(thresholds_dist)),
			"tpr_best_mean": float(np.mean(tprs_best)),
			"fpr_best_mean": float(np.mean(fprs_best)),
			"verification_accuracy_best_mean": float(np.mean(verif_accs_best)),
		}
		summary_rows.append(summary_row)

		summary_per_n[str(n_value)] = {
			"requested_N": n_value,
			"effective_N": n_effective,
			"classification": summary_row,
			"confusion_matrix_mean": cm_mean.tolist(),
			"roc": {
				"fpr_grid": roc_agg["fpr_grid"].tolist(),
				"tpr_mean": roc_agg["tpr_mean"].tolist(),
				"tpr_std": roc_agg["tpr_std"].tolist(),
				"auc_mean": float(roc_agg["auc_mean"]),
				"auc_std": float(roc_agg["auc_std"]),
			},
		}

	metrics_df = pd.DataFrame(metrics_rows)
	metrics_df.to_csv(METRICS_CSV_FILE, index=False)

	save_accuracy_vs_n(summary_rows)
	save_metrics_boxplots(metrics_df)
	save_roc_plot(roc_aggregated_by_n)
	save_intra_inter_distributions(separability_rows)
	save_embedding_projection(person_embeddings, method="pca")
	save_embedding_projection(person_embeddings, method="tsne")

	summary_payload = {
		"metadata": {
			"created_at": datetime.utcnow().isoformat(),
			"dataset_path": DATASET_PATH,
			"model_name": MODEL_NAME,
			"detector_backend": DETECTOR_BACKEND,
			"distance_metric": DISTANCE_METRIC,
			"gallery_aggregation": GALLERY_AGGREGATION,
			"n_values": N_VALUES,
			"repetitions": REPETITIONS,
			"base_seed": BASE_SEED,
			"num_people": len(person_embeddings),
			"images_per_person": {person: len(items) for person, items in person_embeddings.items()},
		},
		"summary_by_n": summary_per_n,
	}

	with open(SUMMARY_JSON_FILE, "w", encoding="utf-8") as f:
		json.dump(summary_payload, f, indent=2)

	print("\nExperimento completado.")
	print(f"CSV: {METRICS_CSV_FILE}")
	print(f"JSON: {SUMMARY_JSON_FILE}")


if __name__ == "__main__":
	run_experiment()
