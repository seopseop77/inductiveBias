import pandas as pd
import sys 

from transform.transform_functions import functions_dict
from config import *
from vit_explainability.vit_visualization_explainability import get_vit_explainability_model, visualize_attention_explainability

def visualize_transform(image_path, target_diameter, transform_fn=None, exist_origin=True):
    if exist_origin:
        transformed_path = transform_fn(image_path)

        model, transform = get_vit_explainability_model()
        visualize_attention_explainability(model, image_path, transform, target_diameter)
        visualize_attention_explainability(model, transformed_path, transform, target_diameter)
    else: 
        model, transform = get_vit_explainability_model()
        visualize_attention_explainability(model, image_path, transform, target_diameter)

if __name__ == "__main__":
    # df = pd.read_csv(LABEL_FILE)
    # sample_row = df.iloc[:3]

    # # 인자로 함수 입력 받아서 사용
    # function_name = sys.argv[1]
    # for _, row in sample_row.iterrows():
    #     visualize_transform(DATA_DIR / row['filename'], row['diameter'], functions_dict[function_name])
    for r in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        transformed_path, diameter = functions_dict["circle"](r)
        visualize_transform(transformed_path, diameter, exist_origin=False)
