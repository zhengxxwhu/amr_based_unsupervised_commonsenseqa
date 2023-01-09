import json
import spacy
from nltk.corpus import stopwords
from tqdm import tqdm


def filter_Q_by_quality_score():
    '''input_path = "pretrained_data/Q_only_based_synthesize_QA_filtered/Q_only_based_synthesize_QA_full_filtered_train.jsonl"
    output_path = "pretrained_data/Q_only_based_synthesize_QA_filtered/Q_only_based_synthesize_QA_full_train_high_quality_0.5.jsonl"'''
    '''input_path = "pretrained_data/Q_only_based_synthesize_QA_filtered/Q_only_based_synthesize_QA_full_filtered_test.jsonl"
    output_path = "pretrained_data/Q_only_based_synthesize_QA_filtered/Q_only_based_synthesize_QA_full_test_high_quality_0.5.jsonl"'''
    '''input_path = "pretrained_data/Q_only_based_synthesize_QA_filtered/Q_only_based_synthesize_QA_full_filtered_train.jsonl"
    output_path = "pretrained_data/Q_only_based_synthesize_QA_filtered/Q_only_based_synthesize_QA_full_train_high_quality_0.3.jsonl"'''
    '''input_path = "pretrained_data/Q_only_based_synthesize_QA_filtered/Q_only_based_synthesize_QA_full_filtered_test.jsonl"
    output_path = "pretrained_data/Q_only_based_synthesize_QA_filtered/Q_only_based_synthesize_QA_full_test_high_quality_0.3.jsonl"'''
    '''input_path = "pretrained_data/Q_only_based_synthesize_QA_filtered/Q_only_based_synthesize_QA_full_filtered_train.jsonl"
    output_path = "pretrained_data/Q_only_based_synthesize_QA_filtered/Q_only_based_synthesize_QA_full_train_high_quality_0.6.jsonl"'''
    '''input_path = "pretrained_data/Q_only_based_synthesize_QA_filtered/Q_only_based_synthesize_QA_full_filtered_test.jsonl"
    output_path = "pretrained_data/Q_only_based_synthesize_QA_filtered/Q_only_based_synthesize_QA_full_test_high_quality_0.6.jsonl"'''
    '''input_path="pretrained_data/Q_only_based_synthesize_with_knowledge_filtered/Q_only_based_synthesize_QA_full_filtered_test.jsonl"
    output_path="pretrained_data/Q_only_based_synthesize_with_knowledge_filtered/Q_only_based_synthesize_QA_full_test_high_quality_0.7.jsonl"'''
    '''input_path="pretrained_data/Q_only_SocialiQA_only_based_synthesize_with_knowledge_filtered/Q_only_based_synthesize_QA_full_filtered_test.jsonl"
    output_path="pretrained_data/Q_only_SocialiQA_only_based_synthesize_with_knowledge_filtered/Q_only_based_synthesize_QA_full_test_high_quality_0.7.jsonl"'''
    '''input_path = "pretrained_data/Q_only_based_synthesize_QA_filtered/Q_only_based_synthesize_QA_full_filtered_train.jsonl"
    output_path = "pretrained_data/Q_only_based_synthesize_QA_filtered/Q_only_based_synthesize_QA_full_train_high_quality_0.4.jsonl"'''
    '''input_path = "pretrained_data/Q_only_based_synthesize_QA_filtered/Q_only_based_synthesize_QA_full_filtered_test.jsonl"
    output_path = "pretrained_data/Q_only_based_synthesize_QA_filtered/Q_only_based_synthesize_QA_full_test_high_quality_0.4.jsonl"'''
    '''input_path = "pretrained_data/Q_only_based_synthesize_QA_filtered/Q_only_based_synthesize_QA_full_filtered_train.jsonl"
    output_path = "pretrained_data/Q_only_based_synthesize_QA_filtered/Q_only_based_synthesize_QA_full_train_high_quality_0.9.jsonl"'''
    input_path = "pretrained_data/Q_only_based_synthesize_QA_filtered/Q_only_based_synthesize_QA_full_filtered_test.jsonl"
    output_path = "pretrained_data/Q_only_based_synthesize_QA_filtered/Q_only_based_synthesize_QA_full_test_high_quality_0.9.jsonl"
    with open(input_path,"r") as input_file, open(output_path,"w") as output_file:
        samples=[json.loads(line) for line in input_file.readlines()]
        for sample in samples:
            #if sample['quality']<0.5:
            #if sample['quality'] < 0.2:
            #if sample['quality'] < 0.3:
            #if sample['quality'] < 0.4:
            #if sample['quality'] < 0.6:
            #if sample['quality'] < 0.7:
            #if sample['quality'] < 0.8:
            if sample['quality'] < 0.9:
                continue
            else:
                output_file.write(json.dumps(sample)+"\n")


if __name__ == "__main__":
    filter_Q_by_quality_score()