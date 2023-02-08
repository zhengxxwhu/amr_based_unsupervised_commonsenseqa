合成问答数据模块    tools

    下载维基百科语料库   https://huggingface.co/datasets/wiki40b

    从维基百科中进行陈述句采样   python tools/sample_wiki.py

    合成问答数据对 python tools/synthesize_question.py

合成数据筛选模块   question_filtering

    训练得到合成问句质量打分模型  python question_filtering/filtering_training_Q_only.py
    
    对合成问句进行质量打分 python question_filtering/synthesize_QA_filtering_Q_only.py --eval_batch_size 500 --synthesize_train_data_path pretrained_data/synthesize_QA_full_files/synthesize_QA_full_train.jsonl --output_dir pretrained_data/Q_only_based_synthesize_QA_filtered --save_path Q_only_based_synthesize_QA_full_filtered_train.jsonl --model_name_or_path model_save/question_filtering_Q_only/roberta-base-cls
    
    基于质量打分筛选数据  python question_filtering/basic_ops.py

训练问答模型模块    training

    将dataset文件转为jsonl文件 training/preprocess.py
    
    对训练数据进行分词及生成及干扰项    python training/tokenized_data.py --hard_candidates --output_dir Q_only_based_synthesize_QA_full_filtered_0.7_with_hard_candidates_roberta --max_seq_length 110 --tokenize_synthesize_QA --train_data_path pretrained_data/Q_only_based_synthesize_QA_filtered/Q_only_based_synthesize_QA_full_train_high_quality_0.7.jsonl --dev_data_path pretrained_data/Q_only_based_synthesize_QA_filtered/Q_only_based_synthesize_QA_full_test_high_quality_0.7.jsonl
    
    训练问答模型  training/train.py --cache_dir Q_only_based_synthesize_QA_full_filtered_0.7_with_hard_candidates_roberta --output_dir model_save/Q_only_based_synthesize_filtered_0.7_with_hard_candidates/roberta-mlm-margin --max_seq_length 110 --save_step 1000 --gradient_accumulation_steps 32 --max_words_to_mask 6 --gpu_id 0 --train_batch_size 1 --max_sequence_per_time 200 --model_type roberta-mlm --model_name_or_path roberta-large --seed 4491

    测试问答模型  training/CQA_evaluate.py --pretrained_model model_save/Q_only_based_synthesize_filtered_0.7_with_hard_candidates/roberta-mlm-margin --max_seq_length 110 --max_sequence_per_time 1000 --model_type roberta-mlm --model_name_or_path roberta-large --vocab_path roberta-large --model_config roberta-large --eval_batch_size 4

基于阈值0.7和0.8的合成数据集文件在pretrained_data/Q_only_based_synthesize_QA_filtered/candidates文件夹下