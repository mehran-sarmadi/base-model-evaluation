CUDA_VISIBLE_DEVICES=6 python3 runner.py \
--model_name_or_path "/mnt/data/ez-workspace/RetroMAE/results/RetroMAE_ourspacedwordpiece_ourlinebyline/checkpoint-1564218"  --language 'fa'

# CUDA_VISIBLE_DEVICES=3 python3 runner.py \
#--model_name_or_path "/mnt/data/mehran-workspace/language-model/tokenizer-evaluation-pipline/train-one-epoch/results/ourdataset_spaced_dataset_50k_word_piece"  --language 'fa'

#CUDA_VISIBLE_DEVICES=5 python3 runner.py \
#--model_name_or_path "/mnt/data/mehran-workspace/language-model/test-fp-16-on-fixed-dataset-our-line-by-line/BERT/train-allfixedfarsi-persian_50k_bbpe_xlmclean_ourdata-mlm-ourlinebyline/checkpoint-3540000" \
#--tokenizer_name_or_path '/mnt/data/ez-workspace/tokenizer/models/persian_50k_bbpe_xlmclean_ourdata/tokenizer' --language 'fa'

#CUDA_VISIBLE_DEVICES=5 python3 runner.py \
#--model_name_or_path "/mnt/data/mehran-workspace/language-model/test-fp-16-on-fixed-dataset-our-line-by-line/BERT/train-allfixedfarsi-persian_50k_bbpe_xlmclean_ourdata-mlm-ourlinebyline/checkpoint-3580000" \
#--tokenizer_name_or_path '/mnt/data/ez-workspace/tokenizer/models/persian_50k_bbpe_xlmclean_ourdata/tokenizer' --language 'fa'

#CUDA_VISIBLE_DEVICES=5 python3 runner.py \
#--model_name_or_path "/mnt/data/mehran-workspace/language-model/test-fp-16-on-fixed-dataset-our-line-by-line/BERT/train-allfixedfarsi-persian_50k_bbpe_xlmclean_ourdata-mlm-ourlinebyline/checkpoint-3600000" \
#--tokenizer_name_or_path '/mnt/data/ez-workspace/tokenizer/models/persian_50k_bbpe_xlmclean_ourdata/tokenizer' --language 'fa'

#CUDA_VISIBLE_DEVICES=5 python3 runner.py \
#--model_name_or_path "/mnt/data/mehran-workspace/language-model/test-fp-16-on-fixed-dataset-our-line-by-line/BERT/train-allfixedfarsi-persian_50k_bbpe_xlmclean_ourdata-mlm-ourlinebyline/checkpoint-3740000" \
#--tokenizer_name_or_path '/mnt/data/ez-workspace/tokenizer/models/persian_50k_bbpe_xlmclean_ourdata/tokenizer' --language 'fa'

#CUDA_VISIBLE_DEVICES=7 python3 runner.py \
#--model_name_or_path "aubmindlab/bert-base-arabert" \
#--language 'ar'

#CUDA_VISIBLE_DEVICES=0 python3 runner.py \
#--model_name_or_path "/mnt/data/mehran-workspace/language-model/tokenizer-evaluation-pipline/BERT-W-Eval/results/multi_50k_bbpe_xlmclean_multidata" \
#--language all
