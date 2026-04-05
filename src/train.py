# ── Simple training (no K-Fold) ──────────────────────────────────────────
train_ds = FloodDataset(train_imgs, train_lbls, transform=train_transform)
val_ds   = FloodDataset(val_imgs,   val_lbls,   transform=val_transform)

train_loader = DataLoader(train_ds, batch_size=CFG.BATCH_SIZE,
                          shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
val_loader   = DataLoader(val_ds,   batch_size=CFG.BATCH_SIZE,
                          shuffle=False, num_workers=2, pin_memory=True)

print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

# ── Train Prithvi ─────────────────────────────────────────────────────────
mA = PrithviSegmentation().to(CFG.DEVICE)
hA = train_model(mA, "Prithvi-100M", "best_prithvi.pth",
                 train_loader, val_loader, lr=5e-5, frozen_epochs=3)
del mA; torch.cuda.empty_cache(); gc.collect()

# ── Train UNet++ B5 ───────────────────────────────────────────────────────
mB = build_smp_model_1().to(CFG.DEVICE)
hB = train_model(mB, "UNet++ B5", "best_smp1.pth",
                 train_loader, val_loader, lr=1.5e-4, frozen_epochs=0)
del mB; torch.cuda.empty_cache(); gc.collect()

# ── Train DeepLabV3+ ──────────────────────────────────────────────────────
mC = build_smp_model_2().to(CFG.DEVICE)
hC = train_model(mC, "DeepLabV3+ R50", "best_smp2.pth",
                 train_loader, val_loader, lr=1.5e-4, frozen_epochs=0)
del mC; torch.cuda.empty_cache(); gc.collect()

print(f"\nPrithvi best FloodIoU : {hA['iou_flood'].max():.4f}")
print(f"UNet++ best FloodIoU  : {hB['iou_flood'].max():.4f}")
print(f"DeepLabV3+ best FloodIoU: {hC['iou_flood'].max():.4f}")
