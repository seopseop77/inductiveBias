import pandas as pd

from transform.transform_boundary import extract_boundary
from config import *
from vit_explainability.vit_visualization_explainability import get_vit_explainability_model, visualize_attention_explainability

def visualize_transform(image_path, target_diameter, transform_fn, exist_origin=True):
    if exist_origin:
        boundary_path = transform_fn(image_path)

        model, transform = get_vit_explainability_model()
        visualize_attention_explainability(model, image_path, transform, target_diameter)
        visualize_attention_explainability(model, boundary_path, transform, target_diameter)
    else: 
        model, transform = get_vit_explainability_model()
        visualize_attention_explainability(model, image_path, transform, target_diameter)

if __name__ == "__main__":
    df = pd.read_csv(LABEL_FILE)
    sample_row = df.iloc[:5]
    for _, row in sample_row.iterrows():
        visualize_transform(DATA_DIR / row['filename'], row['diameter'], extract_boundary)