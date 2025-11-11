#!/usr/bin/env python
"""
Cross-Encoder 모델 학습 스크립트
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments, 
    Trainer
)
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# 상대 경로 import를 위한 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class CrossEncoderDataset(Dataset):
    """
    Cross-Encoder 학습용 데이터셋
    """
    def __init__(self, tokenizer, query_passage_pairs, labels, max_length=512):
        self.tokenizer = tokenizer
        self.query_passage_pairs = query_passage_pairs
        self.labels = labels
        self.max_length = max_length
        
    def __len__(self):
        return len(self.query_passage_pairs)
    
    def __getitem__(self, index):
        query, passage = self.query_passage_pairs[index]
        label = self.labels[index]
        
        # 토큰화
        encoding = self.tokenizer(
            query, 
            passage, 
            padding='max_length',
            truncation='longest_first',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # 배치 차원 제거 (Trainer 사용 시 필요)
        encoding = {key: val.squeeze(0) for key, val in encoding.items()}
        
        # 레이블 추가
        encoding['labels'] = torch.tensor(label, dtype=torch.float)
        
        return encoding

def prepare_training_data(examples_file, negative_ratio=3):
    """
    학습 데이터 준비
    
    Args:
        examples_file: 예제 데이터 파일 경로(JSON)
        negative_ratio: 부정 샘플 비율 (1개의 긍정 샘플당 N개의 부정 샘플)
        
    Returns:
        tuple: (query_passage_pairs, labels)
    """
    # 예제 데이터 로드
    with open(examples_file, 'r', encoding='utf-8') as f:
        examples = json.load(f)
    
    print(f"파인튜닝 데이터 {len(examples)}개 로드됨")
    
    # 모든 파편 수집
    all_fragments = {}
    for example in examples:
        fragment_type = example['fragment_type']
        fragment_path = example['fragment_path']
        fragment_summary = example['fragment_summary']
        
        fragment_id = f"{fragment_type}:{fragment_path}"
        all_fragments[fragment_id] = fragment_summary
    
    # 학습 데이터 준비
    query_passage_pairs = []
    labels = []
    
    for example in examples:
        fragment_type = example['fragment_type']
        fragment_path = example['fragment_path']
        fragment_summary = example['fragment_summary']
        fragment_id = f"{fragment_type}:{fragment_path}"
        questions = example.get('questions', [])
        
        for question in questions:
            # 긍정 예제 (질문-관련 파편)
            query_passage_pairs.append((question, fragment_summary))
            labels.append(1)  # 관련 있음 (긍정)
            
            # 부정 예제 생성 (다른 무관한 파편과 조합)
            negative_fragments = []
            fragments_ids = list(all_fragments.keys())
            np.random.shuffle(fragments_ids)
            
            for other_frag_id in fragments_ids:
                if other_frag_id != fragment_id and len(negative_fragments) < negative_ratio:
                    negative_fragments.append(all_fragments[other_frag_id])
            
            for neg_fragment in negative_fragments:
                query_passage_pairs.append((question, neg_fragment))
                labels.append(0)  # 관련 없음 (부정)
    
    print(f"총 {len(query_passage_pairs)}개 학습 데이터 생성됨")
    print(f"- 긍정 샘플: {labels.count(1)}개")
    print(f"- 부정 샘플: {labels.count(0)}개")
    
    return query_passage_pairs, labels

def train_cross_encoder(examples_file, output_dir='./trained_model', model_name='dragonkue/bge-reranker-v2-m3-ko', epochs=3, batch_size=16):
    """
    Cross-Encoder 모델 학습
    
    Args:
        examples_file: 예제 데이터 파일 경로(JSON)
        output_dir: 모델 저장 디렉토리
        model_name: 기본 모델 이름
        epochs: 학습 에포크 수
        batch_size: 배치 크기
    """
    # 학습 데이터 준비
    query_passage_pairs, labels = prepare_training_data(examples_file)
    
    # 훈련/검증 데이터 분할
    train_pairs, val_pairs, train_labels, val_labels = train_test_split(
        query_passage_pairs, labels, test_size=0.1, random_state=42, stratify=labels
    )
    
    print(f"훈련 데이터: {len(train_pairs)}개")
    print(f"검증 데이터: {len(val_pairs)}개")
    
    # 모델 및 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
    
    # 데이터셋 생성
    train_dataset = CrossEncoderDataset(tokenizer, train_pairs, train_labels)
    val_dataset = CrossEncoderDataset(tokenizer, val_pairs, val_labels)
    
    # 학습 인자 설정
    training_args = TrainingArguments(
		output_dir=output_dir,
		num_train_epochs=epochs,
		per_device_train_batch_size=4,  # 배치 크기 감소
		per_device_eval_batch_size=4,   # 평가 배치 크기도 감소
		gradient_accumulation_steps=4,  # 그래디언트 누적 (효과적으로 16 배치 크기와 유사)
		warmup_steps=50,                # 데이터셋 크기를 고려하여 감소
		weight_decay=0.01,
		logging_dir=f"{output_dir}/logs",
		logging_steps=5,                # 더 자주 로깅
		eval_strategy="steps",          # evaluation_strategy에서 변경
		eval_steps=50,                  # 50 스텝마다 평가
		save_strategy="steps",
		save_steps=50,
		load_best_model_at_end=True,
		metric_for_best_model="loss",
		greater_is_better=False,
		fp16=True,                      # 메모리 절약을 위한 fp16 활성화
		dataloader_num_workers=0,       # 데이터 로더 워커 수 제한
		disable_tqdm=False,             # 진행 상황 표시
    )
    
    # 트레이너 초기화
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    
    # 학습 실행
    print("모델 학습 시작...")
    trainer.train()
    
    # 모델 저장
    print(f"모델 저장 중: {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print("모델 학습 완료!")

def test_cross_encoder(model_dir, examples_file, top_k=3):
    """
    학습된 Cross-Encoder 모델 테스트
    
    Args:
        model_dir: 학습된 모델 디렉토리
        examples_file: 예제 데이터 파일 경로(JSON)
        top_k: 상위 결과 수
    """
    from app.embedding.cross_encoder import CrossEncoder
    
    # 예제 데이터 로드
    with open(examples_file, 'r', encoding='utf-8') as f:
        examples = json.load(f)
    
    # 모든 파편 수집
    all_fragments = []
    for example in examples:
        fragment_type = example['fragment_type']
        fragment_path = example['fragment_path']
        fragment_summary = example['fragment_summary']
        
        all_fragments.append({
            'id': f"{fragment_type}:{fragment_path}",
            'type': fragment_type,
            'file_path': fragment_path,
            'name': fragment_path.split('/')[-1],
            'content_preview': fragment_summary
        })
    
    # Cross-Encoder 모델 로드
    cross_encoder = CrossEncoder(model_name=model_dir)
    
    # 테스트할 질문 준비
    test_questions = []
    for example in examples[:3]:  # 처음 3개 예제에서만 테스트
        for question in example.get('questions', [])[:1]:  # 각 예제에서 1개 질문만 선택
            test_questions.append({
                'question': question,
                'fragment_id': f"{example['fragment_type']}:{example['fragment_path']}"
            })
    
    # 테스트 실행
    print("\n=== Cross-Encoder 테스트 ===")
    for test_case in test_questions:
        question = test_case['question']
        correct_id = test_case['fragment_id']
        
        print(f"\n질문: {question}")
        print(f"정답 파편: {correct_id}")
        
        # 재랭킹 수행
        ranked_results = cross_encoder.rerank(question, all_fragments, top_k=top_k)
        
        print(f"\n상위 {top_k} 결과:")
        for i, result in enumerate(ranked_results):
            print(f"[{i+1}] ID: {result['id']} - 점수: {result['cross_score']:.4f}")
            is_correct = "✓" if result['id'] == correct_id else " "
            print(f"    {is_correct} {result['content_preview'][:100]}...")
        
        print("-" * 60)

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='Cross-Encoder 모델 학습')
    parser.add_argument('--examples', type=str, required=True, help='예제 데이터 파일 경로(JSON)')
    parser.add_argument('--output-dir', type=str, default='./trained_model', help='모델 저장 디렉토리')
    parser.add_argument('--model', type=str, default='dragonkue/bge-reranker-v2-m3-ko', help='기본 모델 이름')
    parser.add_argument('--epochs', type=int, default=3, help='학습 에포크 수')
    parser.add_argument('--batch-size', type=int, default=16, help='배치 크기')
    parser.add_argument('--test', action='store_true', help='학습 후 테스트 수행')
    
    args = parser.parse_args()
    
    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 모델 학습
    train_cross_encoder(
        examples_file=args.examples,
        output_dir=args.output_dir,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    # 테스트
    if args.test:
        test_cross_encoder(args.output_dir, args.examples)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())