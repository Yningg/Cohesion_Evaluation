
nohup  python link_pretrain_exp.py --dataset BTW17 --model_name BTW17 --batch_size 7721 --epochs 100 --dropout 0.1 --hidden_dim 128 --hops 5  --n_heads 8 --n_layers 1 --pe_dim 5 --peak_lr 0.001  --weight_decay=1e-05 --device 0 >> ./logs/BTW17_training.txt 2>&1 &&

python link_pretrain_exp.py --dataset Chicago_COVID --model_name Chicago_COVID --batch_size 4971 --epochs 100 --dropout 0.1 --hidden_dim 128 --hops 5  --n_heads 8 --n_layers 1 --pe_dim 5 --peak_lr 0.001  --weight_decay=1e-05 --device 0 >> ./logs/Chicago_COVID_training.txt 2>&1 &&

python link_pretrain_exp.py --dataset Crawled_Dataset144 --model_name Crawled_Dataset144 --batch_size 10388 --epochs 100 --dropout 0.1 --hidden_dim 128 --hops 5  --n_heads 8 --n_layers 1 --pe_dim 5 --peak_lr 0.001  --weight_decay=1e-05 --device 0 >> ./logs/Crawled_Dataset144_training.txt 2>&1 &&

python link_pretrain_exp.py --dataset Crawled_Dataset26 --model_name Crawled_Dataset26 --batch_size 21236 --epochs 100 --dropout 0.1 --hidden_dim 128 --hops 5  --n_heads 8 --n_layers 1 --pe_dim 5 --peak_lr 0.001  --weight_decay=1e-05 --device 0 >> ./logs/Crawled_Dataset26_training.txt 2>&1 &
