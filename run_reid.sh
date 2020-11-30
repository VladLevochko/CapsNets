spell run \
--mount uploads/celeba:/mnt/data/celeba_np \
--mount uploads/lfw:/mnt/data/lfw_np \
-t T4 \
--tensorboard-dir=src/runs \
"cd src && python train_reid.py \
--batch_size=256 \
--epochs=20 \
--lr=0.1 \
--lamda=1 \
--alpha=0.005 \
--celeba_np_root=/mnt/data/celeba_np \
--lfw_root=/mnt/data/lfw_np \
--model_type=conv \
--pretrained"