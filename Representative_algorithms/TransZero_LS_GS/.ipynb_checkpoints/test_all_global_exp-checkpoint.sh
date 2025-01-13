nohup python accuracy_globalsearch_exp.py --dataset BTW17 >> ./logs/test_all_global_exp.txt 2>&1 &&
python accuracy_globalsearch_exp.py --dataset Chicago_COVID >> ./logs/test_all_global_exp.txt 2>&1 &&
python accuracy_globalsearch_exp.py --dataset Crawled_Dataset144 >> ./logs/test_all_global_exp.txt 2>&1 &&
python accuracy_globalsearch_exp.py --dataset Crawled_Dataset26  >> ./logs/test_all_global_exp.txt 2>&1 &