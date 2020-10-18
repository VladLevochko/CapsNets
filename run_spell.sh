spell run \
-t T4 \
--tensorboard-dir=src/runs \
"cd src && python main.py --experiment=caps --download=True --epochs=50 --lr=0.001"