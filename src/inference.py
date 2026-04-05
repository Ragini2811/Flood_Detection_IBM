# ── 21. Inference ────────────────────────────────────────────────────────────
class TestDataset(Dataset):
    def __init__(self, img_paths):
        self.img_paths = img_paths
    def __len__(self): return len(self.img_paths)
    def __getitem__(self, idx):
        with rasterio.open(self.img_paths[idx]) as f:
            img = f.read().astype(np.float32)
        # Ensure spatial consistency with training
        img = img[:, :CFG.IMG_SIZE, :CFG.IMG_SIZE]
        return torch.tensor(preprocess(img), dtype=torch.float32)

test_ds    = TestDataset(test_imgs)
submission = []
kernel_3   = np.ones((3, 3), np.uint8)
kernel_5   = np.ones((5, 5), np.uint8)

print(f'Inference on {len(test_imgs)} images')
print(f'  Models    : {len(all_fold_models)} architectures')
print(f'  TTA       : {len(CFG.TTA_IDS)} transforms')
print(f'  Threshold : {FLOOD_THRESHOLD:.3f}')

for i, img_path in enumerate(tqdm(test_imgs, desc='Inference')):
    # Extract the preprocessed 6-channel image
    img_np = test_ds[i].numpy()   # (6, H, W)

    # 1. Ensemble logits across all models with TTA
    # all_fold_weights should sum to 1.0 to maintain logit scale
    logits_sum = None
    for model, w in zip(all_fold_models, all_fold_weights):
        logits_tta = predict_with_tta(model, img_np, CFG.TTA_IDS)
        if logits_sum is None:
            logits_sum = w * logits_tta
        else:
            logits_sum += w * logits_tta

    # 2. Softmax to get probabilities for all 3 classes
    # Class 0: No-Flood, Class 1: Flood, Class 2: Water-Body
    probs = torch.softmax(
        torch.tensor(logits_sum, dtype=torch.float32).unsqueeze(0), dim=1
    ).squeeze(0).numpy()
    
    # We focus specifically on Class 1 (Flood) for RLE submission
    flood_prob = probs[1]   # (H, W)

    # 3. Apply the optimized Threshold
    raw_mask = (flood_prob > FLOOD_THRESHOLD).astype(np.uint8)

    # 4. Morphological clean-up to remove small noise and fill gaps
    raw_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN,  kernel_3)
    raw_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, kernel_5)

    # 5. Connected-component filter: remove small islands < 64 pixels
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(raw_mask, connectivity=8)
    clean_mask = np.zeros_like(raw_mask)
    for lbl in range(1, num_labels):
        if stats[lbl, cv2.CC_STAT_AREA] >= 64:
            clean_mask[labels == lbl] = 1

    # 6. RLE Encode: mask_to_rle handles the "0 0" for empty masks internally
    # Ensure this uses Class 1 only as per Phase 2 rules
    rle = rle_encode(clean_mask)
    if not rle or rle.strip() == "":
        rle = "0 0"

    submission.append({'id': test_ids[i], 'rle_mask': rle})

