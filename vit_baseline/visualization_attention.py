import torch 
import cv2 
from PIL import Image
import matplotlib.pyplot as plt
# Attention Map을 가로채기 위한 Hook 설정
attention_weights = None

def hook_fn(module, input, output):
    global attention_weights
    # output 구조: (batch, num_heads, query_tokens, key_tokens)
    attention_weights = output
    
def visualize_attention(model, img_path, transform, target_diameter, IMG_SIZE = 224):
    global attention_weights
    # 이미지 로드 및 변환
    raw_image = Image.open(img_path).convert('RGB')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # 이게 살짝 걸리는데...
    input_tensor = transform(raw_image).unsqueeze(0).to(device)

    # 마지막 블록의 attention 레이어에 hook 등록
    model.blocks[-1].attn.attn_drop.register_forward_hook(hook_fn)

    # 추론
    with torch.no_grad():
        prediction = model(input_tensor)

    # 5. Attention Map 처리
    # attention_weights: [1, num_heads, 197, 197] (197 = 14*14 patches + 1 CLS token)

    nh = attention_weights.shape[1] # 헤드 수

    # [CLS] 토큰(index 0)이 나머지 패치들(index 1~)을 보는 점수만 추출
    # 모든 헤드의 평균을 냅니다.
    attentions = attention_weights[0, :, 0, 1:].reshape(nh, -1)
    attentions = attentions.mean(dim=0) # [196]

    # 14x14 그리드로 재배열
    w_featmap = input_tensor.shape[-2] // model.patch_embed.patch_size[0]
    h_featmap = input_tensor.shape[-1] // model.patch_embed.patch_size[1]
    attentions = attentions.reshape(w_featmap, h_featmap).detach().cpu().numpy()

    # 원래 이미지 크기로 확대
    attentions = cv2.resize(attentions, (IMG_SIZE, IMG_SIZE))

    # 6. 시각화
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 왼쪽: 원본 이미지
    axes[0].imshow(raw_image.resize((IMG_SIZE, IMG_SIZE)))
    axes[0].set_title(f"Original (True Dia: {target_diameter:.2f})")
    axes[0].axis('off')

    # 오른쪽: Attention Overlay
    axes[1].imshow(raw_image.resize((IMG_SIZE, IMG_SIZE)))
    axes[1].imshow(attentions, cmap='jet', alpha=0.5) # 히트맵 오버레이
    axes[1].set_title(f"Attention Map (Pred: {prediction.item():.2f})")
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()