spell hyper grid \
-t T4 \
--param lr=0.00005,0.0001,0.0005,0.001 \
--tensorboard-dir=src/runs \
"cd src && python main.py \
--experiment=conv \
--epochs=50 \
--lr=:lr: \
--data_type=mnist \
--download=True"