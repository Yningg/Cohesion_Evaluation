
nohup  python link_pretrain.py --dataset Chicago_COVID --model_name Chicago_COVID --batch_size 4917 --epochs 100 --dropout 0.1 --hops 5  --n_heads 8 --peak_lr 0.001  --weight_decay=1e-05 --device 0 >> D:/Cohesion_Evaluation/Representative_algorithms/TransZero_LS_GS/logs/Chicago_COVID_training.txt 2>&1 &&

python link_pretrain.py --dataset BTW17 --model_name BTW17 --batch_size 7721 --epochs 100 --dropout 0.1 --hops 5  --n_heads 8 --peak_lr 0.001  --weight_decay=1e-05 --device 0 >> D:/Cohesion_Evaluation/Representative_algorithms/TransZero_LS_GS/logs/BTW17_training.txt 2>&1 &&

python link_pretrain.py --dataset Crawled_Dataset144 --model_name Crawled_Dataset144 --batch_size 10388 --epochs 100 --dropout 0.1 --hops 5  --n_heads 8 --peak_lr 0.001  --weight_decay=1e-05 --device 0 >> D:/Cohesion_Evaluation/Representative_algorithms/TransZero_LS_GS/logs/Crawled_Dataset144_training.txt 2>&1 &&

python link_pretrain.py --dataset Crawled_Dataset26 --model_name Crawled_Dataset26 --batch_size 4000 --epochs 100 --dropout 0.1 --hops 5  --n_heads 8 --peak_lr 0.001  --weight_decay=1e-05 --device 0 >> D:/Cohesion_Evaluation/Representative_algorithms/TransZero_LS_GS/logs/Crawled_Dataset26_training.txt 2>&1 &