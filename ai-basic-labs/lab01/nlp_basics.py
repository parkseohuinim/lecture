"""
NLP 기초 실습
- 토큰화, 불용어 제거, lemmatization
- OpenAI 임베딩 생성 및 코사인 유사도 계산
- 간단한 문장 검색기 구현

실습 항목:
[기초] 1~5번 - API 키만 있으면 바로 실행 가능
1. tiktoken으로 토큰 이해하기 - GPT가 텍스트를 어떻게 보는가
2. NLTK 전처리 파이프라인 - 토큰화, 불용어, 표제어 추출
3. OpenAI 임베딩 생성 - 텍스트를 벡터로 변환
4. 코사인 유사도 계산 - 벡터 간 유사성 측정
5. 간단한 검색 엔진 - 의미 기반 문장 검색

[심화] 6~9번 - 시각화/모델 비교
6. 임베딩 시각화 - t-SNE로 벡터 공간 이해하기 (matplotlib 필요)
7. 오픈소스 임베딩 모델 - Sentence Transformers 소개
8. 임베딩 모델 비교 - small vs large 성능/비용 분석
9. 한글-영어 임베딩 비교 - 다국어 의미 정렬(Alignment) 실험 🆕

실행 모드:
  python nlp_basics.py          # 전체 실습 (기본)
  python nlp_basics.py --demo   # 출력 위주 데모 (API 호출 최소화)
  python nlp_basics.py --run    # 실제 계산 + 시각화 파일 저장
  python nlp_basics.py --quick  # 핵심 실습만 (1~5번)
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Any
import numpy as np
import tiktoken
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from openai import OpenAI
from dotenv import load_dotenv
import ssl

# SSL 인증서 검증 비활성화 (NLTK 다운로드용)
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# 프로젝트 루트의 .env 파일 로드
project_root = Path(__file__).parent.parent
load_dotenv(dotenv_path=project_root / '.env')

# 공통 유틸리티 import를 위한 경로 추가
sys.path.insert(0, str(project_root))
from utils import (
    print_section_header, 
    print_subsection, 
    print_key_points, 
    visualize_similarity_bar,
    cosine_similarity,
    cosine_similarity_normalized,
    is_normalized,
    interpret_cosine_similarity,
    get_openai_client,
    COSINE_THRESHOLDS
)


# ============================================================================
# NLTK 데이터 다운로드
# ============================================================================

def download_nltk_data():
    """필요한 NLTK 데이터 다운로드"""
    print("\n[INFO] NLTK 데이터 확인 중...")
    
    resources = [
        ('tokenizers/punkt_tab', 'punkt_tab'),
        ('corpora/stopwords', 'stopwords'),
        ('corpora/wordnet', 'wordnet'),
        ('corpora/omw-1.4', 'omw-1.4'),
        ('taggers/averaged_perceptron_tagger_eng', 'averaged_perceptron_tagger_eng'),  # POS 태깅용
    ]
    
    download_needed = False
    
    for path, name in resources:
        try:
            nltk.data.find(path)
            print(f"  [OK] '{name}' 이미 설치됨")
        except LookupError:
            print(f"  [~] '{name}' 설치 확인 중...")
            download_needed = True
            try:
                result = nltk.download(name, quiet=True)
                if result:
                    print(f"  [OK] '{name}' 설치 완료")
                else:
                    print(f"  [OK] '{name}' 이미 최신 상태")
            except Exception as e:
                print(f"  [X] '{name}' 설치 실패: {e}")
    
    if not download_needed:
        print("\n[OK] 모든 NLTK 데이터가 이미 준비되어 있습니다!")
    else:
        print("\n[OK] NLTK 데이터 다운로드 완료!")


# ============================================================================
# 1. tiktoken으로 토큰 이해하기
# ============================================================================

def count_tokens_with_tiktoken(text: str, model: str = "gpt-3.5-turbo") -> int:
    """
    tiktoken을 사용하여 텍스트의 토큰 수를 계산
    
    Args:
        text: 토큰 수를 계산할 텍스트
        model: 사용할 모델 이름
    
    Returns:
        토큰 수
    """
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)
    return len(tokens)


def demo_tiktoken():
    """실습 1: tiktoken으로 토큰 이해하기"""
    print("\n" + "="*80)
    print("[1] 실습 1: tiktoken으로 토큰 이해하기")
    print("="*80)
    print("목표: GPT가 텍스트를 어떻게 토큰으로 분해하는지 이해")
    print("핵심: 토큰 != 단어, 한글은 영어보다 더 많은 토큰 사용")
    
    # 토큰이란 무엇인가?
    print_section_header("토큰(Token)이란?", "[INFO]")
    print("""
  ┌─────────────────────────────────────────────────────────┐
  │  [TIP] 토큰의 개념                                       │
  │  ─────────────────────────────────────────────────────  │
  │  • GPT는 텍스트를 '토큰' 단위로 처리합니다               │
  │  • 토큰 != 단어 (단어보다 작거나 클 수 있음)             │
  │  • 영어: 1 단어 = 1~2 토큰                               │
  │  • 한글: 1 글자 = 1.5~3 토큰 (바이트 단위 분해)          │
  │                                                         │
  │  왜 중요한가?                                            │
  │  • API 비용이 토큰 단위로 계산됨                         │
  │  • 컨텍스트 윈도우 제한이 토큰 기준                      │
  │  • 예: GPT-4 Turbo = 128K 토큰 제한                     │
  └─────────────────────────────────────────────────────────┘
    """)
    
    texts = [
        "Hello, how are you?",
        "안녕하세요, 반갑습니다!",
        "This is a longer sentence with more words to demonstrate token counting.",
        "AI와 머신러닝은 현대 기술의 핵심입니다."
    ]
    
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    
    print_section_header("영어 vs 한글 토큰 비교", "[CMP]")
    
    for text in texts:
        token_count = count_tokens_with_tiktoken(text)
        char_count = len(text)
        chars_per_token = char_count / token_count  # 토큰 1개당 문자 수
        
        # 효율성 해석 (토큰당 문자가 많을수록 효율적)
        if chars_per_token >= 4.0:
            efficiency = "매우 효율적"
        elif chars_per_token >= 2.5:
            efficiency = "효율적"
        elif chars_per_token >= 1.5:
            efficiency = "보통"
        else:
            efficiency = "비효율적 (토큰 많이 소모)"
        
        print(f"\n{'─'*60}")
        print(f"텍스트: {text}")
        print(f"문자 수: {char_count}자 | 토큰 수: {token_count}개")
        print(f"토큰당 문자 수: {chars_per_token:.2f}자/토큰 → {efficiency}")
        
        # 실제 토큰 ID 확인
        tokens = encoding.encode(text)
        print(f"\n토큰 ID: {tokens[:10]}{'...' if len(tokens) > 10 else ''}")
        
        # 개별 토큰을 디코딩하고 바이트 정보도 표시
        print(f"\n토큰 분석:")
        for i, token_id in enumerate(tokens[:8]):  # 처음 8개만
            decoded = encoding.decode([token_id])
            byte_repr = encoding.decode_single_token_bytes(token_id)
            
            # 출력 가능한 문자인지 확인하고 적절히 표시
            if decoded.isprintable() and not any(ord(c) > 127 for c in decoded):
                # ASCII 출력 가능 문자
                display = f"'{decoded}'"
            elif all(ord(c) > 127 for c in decoded) and decoded.isprintable():
                # 완전한 유니코드 문자 (한글 등)
                display = f"'{decoded}'"
            else:
                # 불완전한 UTF-8 바이트 시퀀스 (BPE 분해로 인한 부분 바이트)
                display = f"<bytes: {byte_repr.hex()}>"
            
            print(f"  [{i+1}] ID:{token_id:6d} | {display:20s} | raw: {byte_repr}")
        
        if len(tokens) > 8:
            print(f"  ... (나머지 {len(tokens) - 8}개 토큰 생략)")
        
        # 한글 텍스트에 대한 추가 설명
        if any(ord(c) > 127 for c in text):
            # 원본 텍스트의 UTF-8 바이트 표현
            original_bytes = text.encode('utf-8')
            
            print(f"\n  [!] 한글 토큰화 상세 설명:")
            print(f"     원본 UTF-8 바이트: {original_bytes[:30]}{'...' if len(original_bytes) > 30 else ''}")
            print(f"     총 {len(original_bytes)} 바이트 (한글 1글자 = 3바이트)")
            print(f"")
            print(f"     BPE 토큰화 과정:")
            print(f"     - BPE는 바이트 레벨에서 자주 등장하는 패턴을 학습")
            print(f"     - 한글은 영어보다 학습 빈도가 낮아 더 작은 조각으로 분해")
            print(f"     - 예: '안'(U+C548) = b'\\xec\\x95\\x88' → 2개 토큰으로 분해될 수 있음")
            print(f"     - 토큰 1: b'\\xec\\x95' (불완전, 2바이트)")
            print(f"     - 토큰 2: b'\\x88' (나머지 1바이트)")
            print(f"")
            print(f"     [OK] 모든 토큰의 바이트를 연결하면 원본 완벽 복원!")
            print(f"     [!] 개별 토큰은 유효한 문자가 아닐 수 있음 (정상)")
    
    # 핵심 포인트
    print_key_points([
        "- tiktoken: OpenAI 공식 토큰 계산 라이브러리",
        "- 모델마다 다른 인코더 사용 (gpt-3.5-turbo, gpt-4 등)",
        "- 한글은 영어보다 2~3배 더 많은 토큰 소모",
        "- API 비용 추정: 1K 토큰 = $0.001~0.01 (모델별 상이)",
        "- 실무 팁: 긴 한글 문서는 토큰 비용 미리 계산!"
    ], "tiktoken 핵심 포인트")


# ============================================================================
# 2. NLTK 전처리 파이프라인
# ============================================================================

class TextPreprocessor:
    """텍스트 전처리 파이프라인"""
    
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
    
    def tokenize(self, text: str) -> List[str]:
        """텍스트를 토큰으로 분리"""
        return word_tokenize(text.lower())
    
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """불용어 제거"""
        return [token for token in tokens if token not in self.stop_words]
    
    def get_wordnet_pos(self, treebank_tag: str):
        """
        Penn Treebank 품사 태그를 WordNet 품사로 변환
        
        Args:
            treebank_tag: Penn Treebank 형식의 품사 태그
        
        Returns:
            WordNet 품사 상수
        """
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN  # 기본값은 명사
    
    def lemmatize(self, tokens: List[str]) -> List[str]:
        """
        표제어 추출 (lemmatization) - 품사 태깅 포함
        
        Args:
            tokens: 토큰 리스트
        
        Returns:
            기본형으로 변환된 토큰 리스트
        """
        # 품사 태깅 먼저 수행
        pos_tags = pos_tag(tokens)
        
        # 품사 정보를 활용한 lemmatization
        lemmatized = []
        for word, pos in pos_tags:
            wordnet_pos = self.get_wordnet_pos(pos)
            lemma = self.lemmatizer.lemmatize(word, pos=wordnet_pos)
            lemmatized.append(lemma)
        
        return lemmatized
    
    def preprocess(self, text: str, remove_stopwords: bool = True, 
                   lemmatize: bool = True) -> List[str]:
        """
        전체 전처리 파이프라인 실행
        
        Args:
            text: 전처리할 텍스트
            remove_stopwords: 불용어 제거 여부
            lemmatize: 표제어 추출 여부
        
        Returns:
            전처리된 토큰 리스트
        """
        # 1. 토큰화
        tokens = self.tokenize(text)
        
        # 2. 알파벳만 남기기
        tokens = [token for token in tokens if token.isalpha()]
        
        # 3. 불용어 제거
        if remove_stopwords:
            tokens = self.remove_stopwords(tokens)
        
        # 4. 표제어 추출
        if lemmatize:
            tokens = self.lemmatize(tokens)
        
        return tokens


def demo_preprocessing():
    """실습 2: NLTK 전처리 파이프라인"""
    print("\n" + "="*80)
    print("[2] 실습 2: NLTK 전처리 파이프라인")
    print("="*80)
    print("목표: 텍스트 정규화의 필요성과 방법 이해")
    print("핵심: 토큰화 -> 정규화 -> 불용어 제거 -> 표제어 추출")
    
    # 전처리란?
    print_section_header("텍스트 전처리란?", "[DOC]")
    print("""
  ┌─────────────────────────────────────────────────────────┐
  │  [TIP] 왜 전처리가 필요한가?                             │
  │  ─────────────────────────────────────────────────────  │
  │  • "Running", "runs", "ran" -> 모두 "run"의 변형         │
  │  • "the", "is", "a" -> 의미 없는 단어 (불용어)           │
  │  • 대소문자 통일 -> "AI" = "ai" = "Ai"                   │
  │                                                         │
  │  전처리 없이 검색하면?                                   │
  │  • "cats" 검색 시 "cat" 문서 놓침                        │
  │  • "THE CAT" vs "the cat" 다르게 인식                    │
  └─────────────────────────────────────────────────────────┘
    """)
    
    preprocessor = TextPreprocessor()
    
    text = "The cats are running quickly through the beautiful gardens and jumping over fences."
    
    print_section_header("단계별 전처리 과정", "[STEP]")
    print(f"\n원본 텍스트: {text}")
    
    # 1단계: 토큰화
    print_subsection("1단계: 토큰화 (Tokenization)")
    tokens = preprocessor.tokenize(text)
    print(f"  결과: {tokens}")
    print(f"  설명: 문장을 단어 단위로 분리, 소문자 변환")
    
    # 2단계: 알파벳만 남기기
    print_subsection("2단계: 알파벳 필터링")
    alpha_tokens = [token for token in tokens if token.isalpha()]
    print(f"  결과: {alpha_tokens}")
    print(f"  설명: 구두점(., !) 제거")
    
    # 3단계: 불용어 제거
    print_subsection("3단계: 불용어 제거 (Stopword Removal)")
    no_stop = preprocessor.remove_stopwords(alpha_tokens)
    removed = [t for t in alpha_tokens if t not in no_stop]
    print(f"  결과: {no_stop}")
    print(f"  제거됨: {removed}")
    print(f"  설명: 'the', 'are', 'and' 등 의미 없는 단어 제거")
    
    # 불용어 리스트 안내
    stop_words_sample = sorted(list(preprocessor.stop_words))[:15]
    print(f"\n  [INFO] 영어 불용어 (총 {len(preprocessor.stop_words)}개):")
    print(f"  예시: {stop_words_sample}...")
    print(f"  [TIP] 전체 목록: nltk.corpus.stopwords.words('english')")
    
    # 4단계: 표제어 추출 (품사 태깅 포함)
    print_subsection("4단계: 표제어 추출 (Lemmatization + POS 태깅)")
    
    # 품사 태깅 결과 먼저 보여주기
    pos_tags = pos_tag(no_stop)
    print(f"  품사 태깅: {pos_tags}")
    print(f"""
  Penn Treebank 품사 태그 설명:
    • NN/NNS  = 명사 단수/복수 (Noun singular/plural)
    • VB/VBG  = 동사 기본형/현재분사 (Verb base/gerund)
    • VBD/VBN = 동사 과거형/과거분사 (Verb past/past participle)
    • JJ      = 형용사 (Adjective)
    • RB      = 부사 (Adverb)
    • IN      = 전치사 (Preposition)""")
    
    lemmatized = preprocessor.lemmatize(no_stop)
    
    # 변화된 단어 강조
    changes = []
    for orig, lem in zip(no_stop, lemmatized):
        if orig != lem:
            changes.append(f"'{orig}' -> '{lem}'")
    
    print(f"\n  결과: {lemmatized}")
    if changes:
        print(f"  변환됨: {', '.join(changes)}")
    print(f"  설명: 품사에 따라 기본형으로 변환 (동사 running->run, 명사 cats->cat)")
    
    # Lemmatization vs Stemming 비교
    print_section_header("Lemmatization vs Stemming", "[vs]")
    print("""
  ┌─────────────────────────────────────────────────────────┐
  │  [CMP] 비교                                              │
  │  ─────────────────────────────────────────────────────  │
  │  단어        │ Stemming     │ Lemmatization            │
  │  ────────────┼──────────────┼─────────────────────────  │
  │  running     │ runn         │ run (동사 기본형)         │
  │  flies       │ fli          │ fly (동사 기본형)         │
  │  studies     │ studi        │ study (명사 기본형)       │
  │  ────────────┼──────────────┼─────────────────────────  │
  │  특징        │ 빠름, 규칙 기반│ 정확, 사전 기반          │
  │  단점        │ 비문법적 결과  │ 느림, 품사 필요          │
  └─────────────────────────────────────────────────────────┘
  
  [!] 중요: Lemmatization은 품사(POS) 태깅이 필수!
      - 품사 없이 실행하면 모든 단어를 명사로 간주
      - running(동사) → running (변환 안됨) ← 잘못된 결과
      - running(동사) + POS 태깅 → run ← 올바른 결과
      
  [!] Lemmatization의 한계:
      - 비교급/최상급 처리 안됨 (better → good 변환 불가)
      - 불규칙 변화 일부 미지원
      - 완벽한 변환을 위해선 규칙 기반 추가 처리 필요
    """)
    
    # Lemmatization 한계 실제 테스트
    print_subsection("Lemmatization 한계 테스트")
    print("  [실험] 비교급/최상급/불규칙 변화 단어 테스트:\n")
    
    # 테스트할 단어들 (기대값과 함께)
    test_words = [
        ("better", "good", "비교급 → 원급"),
        ("best", "good", "최상급 → 원급"),
        ("worse", "bad", "비교급 → 원급"),
        ("worst", "bad", "최상급 → 원급"),
        ("running", "run", "현재분사 → 기본형"),
        ("ran", "run", "과거형 → 기본형"),
        ("went", "go", "불규칙 과거 → 기본형"),
        ("children", "child", "불규칙 복수 → 단수"),
    ]
    
    print(f"  {'원본':<12} {'기대값':<10} {'실제결과':<12} {'성공여부':<8} 설명")
    print(f"  {'─'*60}")
    
    lemmatizer = preprocessor.lemmatizer
    success_count = 0
    
    for word, expected, description in test_words:
        # 품사 태깅 후 lemmatization
        pos_tags = pos_tag([word])
        wordnet_pos = preprocessor.get_wordnet_pos(pos_tags[0][1])
        result = lemmatizer.lemmatize(word, pos=wordnet_pos)
        
        is_success = result == expected
        if is_success:
            success_count += 1
        
        status = "[v]" if is_success else "[x]"
        print(f"  {word:<12} {expected:<10} {result:<12} {status:<8} {description}")
    
    print(f"\n  [결과] {success_count}/{len(test_words)}개 성공 ({success_count/len(test_words)*100:.0f}%)")
    print(f"""
  [!] 결론: NLTK Lemmatizer는 불규칙 변화에 약합니다!
      - 비교급/최상급: better → good, best → good 변환 불가
      - 일부 불규칙 과거: ran → run 변환 실패
      - went → go는 성공하는 경우 있음 (WordNet 사전에 등록된 경우)
      - 이런 불일치는 WordNet 사전의 커버리지 차이 때문
      
  [TIP] 실무 대안:
      - SpaCy의 lemmatizer (더 정확하고 일관성 있음)
      - 커스텀 매핑 테이블 사용 (비교급/최상급 전용)
      - 또는 임베딩 모델에 맡기기 (전처리 생략)
    """)
    
    # 전체 파이프라인 결과
    print_subsection("전체 파이프라인 결과")
    result = preprocessor.preprocess(text)
    print(f"  원본: {text}")
    print(f"  결과: {result}")
    print(f"  토큰 수: {len(text.split())} -> {len(result)} (약 {(1-len(result)/len(text.split()))*100:.0f}% 감소)")
    
    # 핵심 포인트
    print_key_points([
        "- 토큰화: 텍스트를 의미 단위로 분리",
        "- 불용어 제거: 의미 없는 고빈도 단어 제거 (the, is, a...)",
        "- 표제어 추출: 단어를 사전 기본형으로 변환 (품사 태깅 필수!)",
        "- POS 태깅: 동사/형용사/부사 구분이 있어야 정확한 lemmatization",
        "- 용도: 키워드 추출, BM25 검색, 텍스트 분석"
    ], "전처리 핵심 포인트")
    
    # 중요 주의사항: 검색 방식별 전처리 필요성
    print_section_header("⚠️ 중요: 검색 방식별 전처리 필요성", "[WARN]")
    print("""
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  [!] 초보자가 자주 혼동하는 핵심 포인트!                                 │
  │  ─────────────────────────────────────────────────────────────────────  │
  │                                                                         │
  │  ┌─────────────────────────────────────────────────────────────────┐   │
  │  │  BM25 / 키워드 검색         │  임베딩 기반 검색 (Semantic)      │   │
  │  │  ───────────────────────────┼──────────────────────────────────│   │
  │  │  전처리 필수! ✓             │  전처리 불필요 (오히려 해로움!) ✗ │   │
  │  │                             │                                   │   │
  │  │  이유:                      │  이유:                            │   │
  │  │  • 정확한 단어 매칭 필요    │  • 임베딩 모델이 문맥 파악        │   │
  │  │  • "cats" ≠ "cat" (다른 단어)│  • 원문 그대로가 의미 보존       │   │
  │  │  • 불용어가 노이즈로 작용   │  • 전처리 시 의미 손실 가능!      │   │
  │  │                             │                                   │   │
  │  │  예시:                      │  예시:                            │   │
  │  │  "The cats are running"    │  "The cats are running"           │   │
  │  │  → ["cat", "run"]          │  → 그대로 임베딩 생성             │   │
  │  │  (전처리 후 검색)           │  (원문 그대로 검색)               │   │
  │  └─────────────────────────────┴──────────────────────────────────┘   │
  │                                                                         │
  │  [!] 실수 사례:                                                         │
  │  ─────────────────────────────────────────────────────────────────────  │
  │  "임베딩 검색하려고 전처리했더니 검색 품질이 떨어졌어요"                │
  │  → 원인: 불용어 제거로 "not", "no" 같은 부정어도 제거됨                 │
  │  → "I love this" vs "I do not love this" 구분 불가!                    │
  │                                                                         │
  │  [결론]                                                                 │
  │  ─────────────────────────────────────────────────────────────────────  │
  │  • BM25/TF-IDF 검색 → 전처리 필수 (lab03 Hybrid 검색에서 사용)         │
  │  • OpenAI/Sentence Transformers 임베딩 → 전처리 하지 마세요!           │
  │  • 이 실습은 전처리의 "개념"을 이해하기 위한 것 (임베딩에 적용 X)       │
  └─────────────────────────────────────────────────────────────────────────┘
    """)


# ============================================================================
# 3. OpenAI 임베딩 생성
# ============================================================================

class EmbeddingGenerator:
    """OpenAI 임베딩 생성기"""
    
    def __init__(self, api_key: str = None):
        # 공통 헬퍼 사용 (SSL 인증서 검증 우회 포함)
        self.client = get_openai_client(api_key)
        self.model = "text-embedding-3-small"
    
    def get_embedding(self, text: str) -> List[float]:
        """
        단일 텍스트의 임베딩 생성
        
        Args:
            text: 임베딩을 생성할 텍스트
        
        Returns:
            임베딩 벡터
        
        Raises:
            Exception: API 호출 실패 시
        """
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"\n[!] 임베딩 생성 실패: {e}")
            print(f"[TIP] 확인 사항:")
            print(f"     1. OPENAI_API_KEY가 올바른지 확인")
            print(f"     2. 네트워크 연결 상태 확인")
            print(f"     3. API 사용량 한도 확인")
            raise
    
    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        여러 텍스트의 임베딩을 배치로 생성
        
        Args:
            texts: 임베딩을 생성할 텍스트 리스트
        
        Returns:
            임베딩 벡터 리스트
        
        Raises:
            Exception: API 호출 실패 시
        """
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=texts
            )
            return [data.embedding for data in response.data]
        except Exception as e:
            print(f"\n[!] 배치 임베딩 생성 실패: {e}")
            print(f"[TIP] 확인 사항:")
            print(f"     1. OPENAI_API_KEY가 올바른지 확인")
            print(f"     2. 네트워크 연결 상태 확인")
            print(f"     3. 배치 크기가 너무 크지 않은지 확인 (최대 2048개)")
            raise


def demo_embeddings():
    """실습 3: OpenAI 임베딩 생성"""
    print("\n" + "="*80)
    print("[3] 실습 3: OpenAI 임베딩 생성")
    print("="*80)
    print("목표: 텍스트가 어떻게 숫자 벡터로 변환되는지 이해")
    print("핵심: 의미가 비슷한 텍스트 -> 비슷한 벡터 -> 가까운 거리")
    
    # 임베딩이란?
    print_section_header("임베딩(Embedding)이란?", "[INFO]")
    print("""
  ┌─────────────────────────────────────────────────────────┐
  │  [TIP] 임베딩의 개념                                     │
  │  ─────────────────────────────────────────────────────  │
  │  • 텍스트 -> 고정 길이 숫자 벡터로 변환                  │
  │  • 예: "고양이" -> [0.1, -0.3, 0.5, ..., 0.2] (1536차원) │
  │                                                         │
  │  왜 벡터로 변환하는가?                                   │
  │  • 컴퓨터는 숫자만 연산 가능                             │
  │  • 벡터 공간에서 의미적 유사성 측정 가능                 │
  │  • "왕 - 남자 + 여자 = 여왕" 같은 연산 가능             │
  │                                                         │
  │  OpenAI 임베딩 모델:                                     │
  │  • text-embedding-3-small: 1536차원, 저렴, 빠름          │
  │  • text-embedding-3-large: 3072차원, 고성능              │
  │                                                         │
  │  [TIP] 모델 선택 가이드:                                  │
  │  • small: 검색/RAG 대부분의 용도에 충분                  │
  │  • large: 법률/의학/논문급 의미 정밀도가 필요할 때만     │
  │           (비용 2배, 성능 향상은 도메인에 따라 5~15%)    │
  └─────────────────────────────────────────────────────────┘
    """)
    
    # API 키 확인
    if not os.getenv("OPENAI_API_KEY"):
        print("\n[!] OPENAI_API_KEY 환경변수를 설정해주세요!")
        return
    
    generator = EmbeddingGenerator()
    
    # 단일 임베딩
    print_section_header("단일 텍스트 임베딩", "[DOC]")
    
    text = "Artificial intelligence is transforming the world."
    embedding = generator.get_embedding(text)
    
    print(f"\n텍스트: '{text}'")
    print(f"\n임베딩 결과:")
    print(f"  • 벡터 차원: {len(embedding)}")
    print(f"  • 처음 5개 값: {[round(v, 4) for v in embedding[:5]]}")
    print(f"  • 마지막 5개 값: {[round(v, 4) for v in embedding[-5:]]}")
    print(f"  • 값의 범위: [{min(embedding):.4f}, {max(embedding):.4f}]")
    
    # 벡터 시각화 (간단한 히스토그램)
    print(f"\n  값 분포 시각화:")
    bins = [0, 0, 0, 0, 0]  # -0.1~-0.05, -0.05~0, 0~0.05, 0.05~0.1, 기타
    for v in embedding:
        if v < -0.05:
            bins[0] += 1
        elif v < 0:
            bins[1] += 1
        elif v < 0.05:
            bins[2] += 1
        elif v < 0.1:
            bins[3] += 1
        else:
            bins[4] += 1
    
    labels = ["< -0.05", "-0.05~0", "0~0.05", "0.05~0.1", "> 0.1"]
    max_bin = max(bins)
    for label, count in zip(labels, bins):
        bar_len = int(count / max_bin * 30)
        print(f"    {label:>10}: {'#' * bar_len} ({count})")
    
    # L2 노름 계산
    l2_norm = np.sqrt(sum(v**2 for v in embedding))
    
    # OpenAI 임베딩 특성 설명
    print(f"""
  [!] OpenAI 임베딩 특성:
     • L2 정규화됨: 벡터 크기(L2 노름) = {l2_norm:.4f} (≈ 1.0)
     • 대부분 값이 -0.1 ~ 0.1 사이에 분포
     • {len(embedding)}차원이므로 개별 값은 0에 가까움
     • 코사인 유사도 계산에 최적화된 형태
     • 정규화 덕분에 내적(dot product)만으로 유사도 계산 가능""")
    
    # 배치 임베딩
    print_section_header("배치 임베딩 (효율적인 방법)", "[BATCH]")
    
    texts = [
        "I love machine learning.",
        "Deep learning is a subset of AI.",
        "Python is a great programming language."
    ]
    
    print("\n[DOC] 임베딩 생성 코드:")
    print("  ┌─────────────────────────────────────────────────────")
    print("  │ # 비효율적: 개별 호출 (3번 API 호출, 지연 시간 3배)")
    print("  │ embeddings = []")
    print("  │ for text in texts:")
    print("  │     response = client.embeddings.create(input=text)")
    print("  │     embeddings.append(response.data[0].embedding)")
    print("  │")
    print("  │ # 효율적: 배치 호출 (1번 API 호출, 비용 동일)")
    print("  │ response = client.embeddings.create(input=texts)")
    print("  │ embeddings = [data.embedding for data in response.data]")
    print("  └─────────────────────────────────────────────────────")
    
    embeddings = generator.get_embeddings_batch(texts)
    
    print(f"\n배치 임베딩 결과:")
    for i, (text, emb) in enumerate(zip(texts, embeddings)):
        print(f"  {i+1}. '{text}'")
        print(f"     차원: {len(emb)}, 처음 5개: {[round(v, 4) for v in emb[:5]]}")
    
    # 핵심 포인트
    print_key_points([
        "- 임베딩: 텍스트 -> 고차원 벡터 (의미를 숫자로 인코딩)",
        "- text-embedding-3-small: 1536차원, 검색/RAG 대부분에 충분",
        "- text-embedding-3-large: 법률/의학/논문급 정밀도 필요시만 사용",
        "- 배치 처리: 여러 텍스트를 한 번에 -> API 호출 최소화, 비용 절약",
        "- 비용: small ~$0.00002/1K토큰, large ~$0.00013/1K토큰",
        "- 용도: 유사도 검색, 클러스터링, 분류, RAG"
    ], "임베딩 핵심 포인트")


# ============================================================================
# 4. 코사인 유사도 계산 (utils.py에서 import한 함수 사용)
# ============================================================================
# Note: cosine_similarity, cosine_similarity_normalized 함수는 utils.py에 정의되어 있습니다.
# from utils import cosine_similarity, cosine_similarity_normalized, is_normalized


def one_to_many_similarity(query_embedding: List[float], 
                          document_embeddings: List[List[float]]) -> List[float]:
    """
    1:N 유사도 계산 (하나의 쿼리와 여러 문서)
    
    Args:
        query_embedding: 쿼리 임베딩
        document_embeddings: 문서 임베딩 리스트
    
    Returns:
        각 문서와의 유사도 리스트
    """
    similarities = []
    for doc_emb in document_embeddings:
        sim = cosine_similarity(query_embedding, doc_emb)
        similarities.append(sim)
    return similarities


def many_to_many_similarity(embeddings1: List[List[float]], 
                           embeddings2: List[List[float]]) -> np.ndarray:
    """
    N:M 유사도 계산 (여러 쿼리와 여러 문서)
    
    Args:
        embeddings1: 첫 번째 임베딩 리스트
        embeddings2: 두 번째 임베딩 리스트
    
    Returns:
        유사도 행렬 (N x M)
    """
    matrix = np.zeros((len(embeddings1), len(embeddings2)))
    
    for i, emb1 in enumerate(embeddings1):
        for j, emb2 in enumerate(embeddings2):
            matrix[i][j] = cosine_similarity(emb1, emb2)
    
    return matrix


def demo_similarity():
    """실습 4: 코사인 유사도 계산"""
    print("\n" + "="*80)
    print("[4] 실습 4: 코사인 유사도 계산")
    print("="*80)
    print("목표: 벡터 간 유사성을 측정하는 방법 이해")
    print("핵심: 코사인 유사도 = 벡터 방향의 유사성 (크기 무관)")
    
    # 코사인 유사도란?
    print_section_header("코사인 유사도란?", "[INFO]")
    print("""
  ┌─────────────────────────────────────────────────────────┐
  │  [TIP] 코사인 유사도 공식                                │
  │  ─────────────────────────────────────────────────────  │
  │                                                         │
  │              A · B           Σ(Aᵢ × Bᵢ)                 │
  │   cos θ = ───────────  =  ─────────────────            │
  │            |A| × |B|      √(ΣAᵢ²) × √(ΣBᵢ²)            │
  │                                                         │
  │  구성 요소:                                              │
  │  • A · B   : 두 벡터의 내적 (dot product)               │
  │  • |A|, |B|: 벡터의 L2 노름 (크기, magnitude)           │
  │  • θ      : 두 벡터 사이의 각도                         │
  │                                                         │
  │  값의 범위 (-1 ~ +1):                                    │
  │  • +1 : 완전히 같은 방향 (매우 유사)                     │
  │  •  0 : 직각 (관련 없음)                                │
  │  • -1 : 반대 방향 (실제 임베딩에서는 거의 없음)          │
  │                                                         │
  │  [TIP] 실무 해석 기준:                                   │
  │  • 0.8+ : 매우 유사 (거의 같은 의미)                     │
  │  • 0.6~0.8 : 관련 있음                                  │
  │  • 0.4~0.6 : 약간 관련                                  │
  │  • 0.4 미만 : 다른 주제                                 │
  └─────────────────────────────────────────────────────────┘
    """)
    
    if not os.getenv("OPENAI_API_KEY"):
        print("\n[!] OPENAI_API_KEY 환경변수를 설정해주세요!")
        return
    
    generator = EmbeddingGenerator()
    
    # 문장 준비
    sentences = [
        "I love programming in Python.",
        "Python is my favorite programming language.",
        "I enjoy cooking Italian food.",
        "Machine learning is fascinating.",
    ]
    
    # 임베딩 생성
    embeddings = generator.get_embeddings_batch(sentences)
    
    # 1:N 유사도 계산
    print_section_header("1:N 유사도 계산", "[>>>]")
    
    query = "I like coding with Python."
    query_embedding = generator.get_embedding(query)
    
    print(f"\n쿼리: '{query}'")
    print(f"\n각 문장과의 코사인 유사도:")
    print(f"{'─'*60}")
    
    similarities = one_to_many_similarity(query_embedding, embeddings)
    
    # 결과를 유사도 순으로 정렬하여 표시
    sorted_results = sorted(zip(sentences, similarities), key=lambda x: x[1], reverse=True)
    
    for sentence, sim in sorted_results:
        bar = visualize_similarity_bar(sim, 30)
        
        # 유사도 해석 (utils.py의 공통 함수 사용)
        interpretation = interpret_cosine_similarity(sim)
        
        print(f"\n  {bar} {sim:.4f} {interpretation}")
        print(f"  '{sentence}'")
    
    # 가장 유사한 문장
    most_similar_idx = np.argmax(similarities)
    print(f"\n[#1] 가장 유사한 문장: '{sentences[most_similar_idx]}'")
    print(f"     유사도: {similarities[most_similar_idx]:.4f}")
    
    # 정규화된 벡터 내적 = 코사인 유사도 실증
    print_section_header("정규화된 벡터: 내적 = 코사인 유사도", "[MATH]")
    
    print("""
  ┌─────────────────────────────────────────────────────────┐
  │  [TIP] OpenAI 임베딩의 비밀                              │
  │  ─────────────────────────────────────────────────────  │
  │  OpenAI 임베딩은 L2 정규화되어 있습니다.                  │
  │  즉, ||A|| = ||B|| = 1.0                                │
  │                                                         │
  │  따라서 코사인 유사도 공식이 단순해집니다:                │
  │                                                         │
  │       A · B           A · B                             │
  │   ─────────────  =  ─────────  =  A · B                 │
  │    |A| × |B|         1 × 1                              │
  │                                                         │
  │  결론: 정규화된 벡터에서 내적만으로 유사도 계산 가능!     │
  │        (나눗셈 연산 생략 → 더 빠른 계산)                  │
  └─────────────────────────────────────────────────────────┘
    """)
    
    # 실제로 검증
    print("[실험] 정규화 여부 및 내적 vs 코사인 유사도 비교:")
    print(f"{'─'*60}")
    
    # 쿼리 임베딩 정규화 확인
    query_norm = np.linalg.norm(query_embedding)
    print(f"\n  쿼리 벡터 L2 노름: {query_norm:.6f}")
    print(f"  정규화 여부: {'[v] 정규화됨 (노름 ≈ 1.0)' if is_normalized(query_embedding) else '[x] 정규화 안됨'}")
    
    # 첫 번째 문서 임베딩 정규화 확인
    doc_norm = np.linalg.norm(embeddings[0])
    print(f"\n  문서 벡터 L2 노름: {doc_norm:.6f}")
    print(f"  정규화 여부: {'[v] 정규화됨 (노름 ≈ 1.0)' if is_normalized(embeddings[0]) else '[x] 정규화 안됨'}")
    
    # 내적 vs 코사인 유사도 비교
    print(f"\n  계산 방법 비교:")
    
    # 방법 1: 전체 공식 (나눗셈 포함)
    sim_full = cosine_similarity(query_embedding, embeddings[0])
    
    # 방법 2: 내적만 (정규화된 벡터용)
    sim_dot = cosine_similarity_normalized(query_embedding, embeddings[0])
    
    print(f"    방법 1 (전체 공식): {sim_full:.10f}")
    print(f"    방법 2 (내적만):    {sim_dot:.10f}")
    print(f"    차이:               {abs(sim_full - sim_dot):.2e}")
    
    if abs(sim_full - sim_dot) < 1e-6:
        print(f"\n  [v] 결과가 동일합니다! 정규화된 벡터에서는 내적만으로 충분합니다.")
        print(f"      → 실무에서 대용량 검색 시 계산 효율이 중요합니다.")
    
    # N:M 유사도 계산
    print_section_header("N:M 유사도 행렬", "[INFO]")
    
    queries = [
        "Programming languages",
        "Food and cooking"
    ]
    query_embeddings = generator.get_embeddings_batch(queries)
    
    similarity_matrix = many_to_many_similarity(query_embeddings, embeddings)
    
    print("\n유사도 행렬:")
    print(f"{'─'*80}")
    
    # 헤더 출력
    print(f"{'쿼리 \\ 문서':<20}", end="")
    for i in range(len(sentences)):
        print(f"Doc{i+1:2d}  ", end="")
    print()
    print(f"{'─'*80}")
    
    # 각 쿼리별 유사도 출력
    for i, query in enumerate(queries):
        print(f"{query:<20}", end="")
        for j in range(len(sentences)):
            score = similarity_matrix[i][j]
            # 높은 점수 강조
            if score >= 0.5:
                print(f"[{score:.4f}]", end="")
            else:
                print(f" {score:.4f} ", end="")
        print()
    
    print(f"{'─'*80}")
    
    # 문서 목록 출력
    print("\n문서 목록:")
    for i, sentence in enumerate(sentences):
        print(f"  Doc{i+1:2d}: {sentence}")
    
    # 각 쿼리별 가장 유사한 문서
    print("\n[*] 각 쿼리별 가장 유사한 문서:")
    for i, query in enumerate(queries):
        most_similar_idx = np.argmax(similarity_matrix[i])
        score = similarity_matrix[i][most_similar_idx]
        print(f"  '{query}'")
        print(f"    -> Doc{most_similar_idx+1}: '{sentences[most_similar_idx]}' ({score:.4f})")
    
    # 핵심 포인트
    print_key_points([
        "- 코사인 유사도: 벡터 방향의 유사성 측정 (-1 ~ 1)",
        "- 임베딩에서: 0.8+ (매우 유사), 0.6~0.8 (관련), 0.4~0.6 (약간 관련), 0.4- (다른 주제)",
        "- 1:N 검색: 쿼리 vs 모든 문서 -> 가장 유사한 문서 찾기",
        "- N:M 검색: 여러 쿼리 vs 여러 문서 -> 행렬 형태 결과",
        "- 실무 팁: Vector DB는 내부적으로 이 계산을 최적화"
    ], "유사도 계산 핵심 포인트")


# ============================================================================
# 5. 간단한 검색 엔진
# ============================================================================

class SimpleSearchEngine:
    """간단한 의미 기반 검색 엔진 (numpy 최적화)"""
    
    def __init__(self, api_key: str = None):
        self.generator = EmbeddingGenerator(api_key)
        self.documents: List[str] = []
        self.embeddings: np.ndarray = None  # numpy array로 저장 (메모리 효율)
    
    def add_documents(self, documents: List[str]):
        """
        문서들을 검색 엔진에 추가
        
        Args:
            documents: 추가할 문서 리스트
        """
        print(f"\n{len(documents)}개의 문서를 인덱싱 중...")
        self.documents.extend(documents)
        new_embeddings = self.generator.get_embeddings_batch(documents)
        
        # numpy array로 변환하여 저장 (메모리 효율 + 연산 속도 향상)
        new_embeddings_array = np.array(new_embeddings)
        if self.embeddings is None:
            self.embeddings = new_embeddings_array
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings_array])
        
        print(f"인덱싱 완료! ({self.embeddings.shape[0]}개 문서 × {self.embeddings.shape[1]}차원 벡터)")
    
    def search(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        쿼리와 가장 유사한 문서 검색
        
        Args:
            query: 검색 쿼리
            top_k: 반환할 상위 결과 개수
        
        Returns:
            (문서, 유사도) 튜플의 리스트
        """
        if not self.documents or self.embeddings is None:
            return []
        
        # 쿼리 임베딩 생성
        query_embedding = np.array(self.generator.get_embedding(query))
        
        # numpy 벡터화 연산으로 유사도 계산 (더 빠름)
        # 코사인 유사도 = dot product / (norm1 * norm2)
        # OpenAI 임베딩은 이미 정규화되어 있으므로 dot product만으로 계산 가능
        similarities = np.dot(self.embeddings, query_embedding)
        
        # 상위 k개 결과 추출
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append((self.documents[idx], float(similarities[idx])))
        
        return results
    
    def print_search_results(self, query: str, top_k: int = 3):
        """검색 결과를 보기 좋게 출력"""
        print(f"\n[>>>] 검색 쿼리: '{query}'")
        print("─" * 60)
        
        results = self.search(query, top_k)
        
        if not results:
            print("검색 결과가 없습니다.")
            return
        
        print(f"\n상위 {len(results)}개 결과:\n")
        for i, (doc, score) in enumerate(results, 1):
            bar = visualize_similarity_bar(score, 25)
            
            # 점수 해석 (실무 기준과 일관되게)
            if score >= 0.8:
                interpretation = "[v] 매우 유사"
            elif score >= 0.6:
                interpretation = "[~] 관련 있음"
            elif score >= 0.4:
                interpretation = "[o] 약간 관련"
            else:
                interpretation = "[x] 다른 주제"
            
            print(f"[{i}] {bar} {score:.4f} {interpretation}")
            print(f"    {doc}\n")


def demo_search_engine():
    """실습 5: 간단한 검색 엔진"""
    print("\n" + "="*80)
    print("[5] 실습 5: 간단한 검색 엔진 (의미 기반)")
    print("="*80)
    print("목표: 임베딩 기반 검색 시스템의 작동 원리 이해")
    print("핵심: 문서 인덱싱 -> 쿼리 임베딩 -> 유사도 검색 -> 순위 정렬")
    
    # 검색 엔진 구조
    print_section_header("의미 기반 검색 엔진 구조", "[ARCH]")
    print("""
  ┌─────────────────────────────────────────────────────────┐
  │  [STEP 1] 인덱싱 단계 (오프라인)                         │
  │  ─────────────────────────────────────────────────────  │
  │  문서들 -> 임베딩 생성 -> 벡터 저장                       │
  │                                                         │
  │  [STEP 2] 검색 단계 (온라인)                             │
  │  ─────────────────────────────────────────────────────  │
  │  1. 쿼리 입력                                           │
  │  2. 쿼리 임베딩 생성                                    │
  │  3. 저장된 벡터들과 유사도 계산                          │
  │  4. 상위 k개 결과 반환                                  │
  │                                                         │
  │  [!] 이것이 RAG의 "Retrieval" 부분!                      │
  └─────────────────────────────────────────────────────────┘
    """)
    
    if not os.getenv("OPENAI_API_KEY"):
        print("\n[!] OPENAI_API_KEY 환경변수를 설정해주세요!")
        return
    
    # 검색 엔진 초기화
    search_engine = SimpleSearchEngine()
    
    # 샘플 문서 추가
    print_section_header("문서 인덱싱", "[LIST]")
    
    documents = [
        "Python is a high-level programming language known for its simplicity.",
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning uses neural networks with multiple layers.",
        "Natural language processing helps computers understand human language.",
        "Computer vision enables machines to interpret visual information.",
        "Data science involves extracting insights from data.",
        "JavaScript is commonly used for web development.",
        "SQL is used for managing relational databases.",
        "Cloud computing provides on-demand computing resources.",
        "Cybersecurity protects systems from digital attacks.",
        "The weather is beautiful today with clear skies.",
        "I love eating pizza and pasta for dinner.",
        "Exercise and healthy eating are important for wellness.",
        "Traveling to new places broadens your perspective.",
        "Reading books is a great way to learn new things.",
    ]
    
    print("\n인덱싱할 문서:")
    for i, doc in enumerate(documents[:5], 1):
        print(f"  {i}. {doc}")
    print(f"  ... ({len(documents) - 5}개 더)")
    
    search_engine.add_documents(documents)
    print(f"\n[OK] 총 {len(documents)}개 문서 인덱싱 완료")
    
    # 다양한 쿼리로 검색
    print_section_header("검색 테스트", "[>>>]")
    
    queries = [
        "What is AI and machine learning?",
        "Tell me about programming languages",
        "How can I stay healthy?",
        "I want to learn about databases",
    ]
    
    for query in queries:
        search_engine.print_search_results(query, top_k=3)
    
    # 검색 쿼리 작성 팁
    print_section_header("검색 쿼리 작성 팁", "[TIP]")
    print("""
  ┌─────────────────────────────────────────────────────────┐
  │  [!] 쿼리 표현이 유사도에 영향을 줍니다!                  │
  │  ─────────────────────────────────────────────────────  │
  │                                                         │
  │  나쁜 예: "Tell me about programming languages"         │
  │    → 불필요한 단어(Tell me about)가 유사도를 낮춤       │
  │    → 결과: 0.38 (낮은 관련성)                           │
  │                                                         │
  │  좋은 예: "programming languages"                       │
  │    → 핵심 키워드만 사용하면 유사도 향상                  │
  │    → 결과: 0.55+ (중간 관련성)                          │
  │                                                         │
  │  [TIP] 의미 검색이라도 쿼리는 간결하게!                  │
  │  [TIP] "What is", "Tell me about" 등은 노이즈           │
  └─────────────────────────────────────────────────────────┘
    """)
    
    # 키워드 검색 vs 의미 검색 비교
    print_section_header("키워드 검색 vs 의미 검색", "[vs]")
    print("""
  ┌─────────────────────────────────────────────────────────┐
  │  [CMP] 비교                                              │
  │  ─────────────────────────────────────────────────────  │
  │  키워드 검색 (BM25)         │ 의미 검색 (임베딩)          │
  │  ───────────────────────────┼────────────────────────── │
  │  "Python" 검색 시           │ "Python" 검색 시           │
  │  -> "Python" 포함 문서만    │ -> 프로그래밍 관련 문서도  │
  │                             │    (JavaScript, SQL 등)    │
  │  ───────────────────────────┼────────────────────────── │
  │  장점: 빠름, 정확한 키워드  │ 장점: 동의어, 유사 개념    │
  │  단점: 동의어 못 찾음       │ 단점: 임베딩 비용 필요     │
  │  ───────────────────────────┼────────────────────────── │
  │  [TIP] 실무: 둘을 결합한 Hybrid 검색 사용 (lab03 학습)  │
  └─────────────────────────────────────────────────────────┘
    """)
    
    # 현재 방식의 한계와 Vector DB 필요성
    print_section_header("현재 방식의 한계", "[!]")
    print(f"""
  ┌─────────────────────────────────────────────────────────┐
  │  [!] 선형 검색의 시간 복잡도: O(n)                       │
  │  ─────────────────────────────────────────────────────  │
  │  현재 인덱싱된 문서: {len(search_engine.documents)}개                              │
  │                                                         │
  │  문서 수에 따른 검색 시간 (O(n) 선형 증가):              │
  │  • 1,000개    → ~10ms   (실시간 가능)                   │
  │  • 10,000개   → ~100ms  (약간 지연)                     │
  │  • 100,000개  → ~1초    (느림)                          │
  │  • 1,000,000개 → ~10초   (실시간 불가!)                 │
  │  (※ 실제 시간은 하드웨어/환경에 따라 다름)               │
  │                                                         │
  │  [>>>] Vector DB (ChromaDB, Pinecone 등)의 해결책:      │
  │  • ANN (Approximate Nearest Neighbor) 알고리즘 사용     │
  │  • 시간 복잡도: O(log n)                                │
  │  • 1,000,000개 → ~10ms (1000배 빠름!)                   │
  │  • 약간의 정확도 trade-off (보통 95%+ 정확도)           │
  │                                                         │
  │  → lab02에서 ChromaDB로 대용량 검색 학습                │
  └─────────────────────────────────────────────────────────┘
    """)
    
    # 핵심 포인트
    print_key_points([
        "- 의미 검색: 키워드가 달라도 의미가 비슷하면 검색됨",
        "- 쿼리 팁: 불필요한 표현 제거, 핵심 키워드만 사용",
        "- 인덱싱: 문서를 임베딩으로 변환하여 저장 (1회)",
        "- 검색: 쿼리 임베딩 -> 유사도 계산 -> 순위 정렬",
        "- 한계: O(n) 복잡도 -> Vector DB(O(log n))로 해결 (lab02)",
        "- 발전: RAG = 검색 + LLM 답변 생성 (lab03)"
    ], "검색 엔진 핵심 포인트")


# ============================================================================
# 6. 임베딩 시각화
# ============================================================================

def demo_embedding_visualization():
    """실습 6: 임베딩 시각화 - t-SNE로 벡터 공간 이해하기"""
    print("\n" + "="*80)
    print("[6] 실습 6: 임베딩 시각화 - t-SNE로 벡터 공간 이해하기")
    print("="*80)
    print("목표: 고차원 임베딩을 2D로 시각화하여 의미적 클러스터 확인")
    print("핵심: 비슷한 의미의 텍스트는 시각화에서도 가까이 모임")
    
    # t-SNE 개념 설명
    print_section_header("차원 축소란?", "[INFO]")
    print("""
  ┌─────────────────────────────────────────────────────────┐
  │  [!] 문제: 1536차원을 어떻게 이해할까?                   │
  │  ─────────────────────────────────────────────────────  │
  │                                                         │
  │  임베딩 벡터: [0.1, -0.3, 0.5, ..., 0.2]  ← 1536개 숫자 │
  │  → 사람이 직접 해석하기 불가능                          │
  │  → 2D/3D로 변환하여 시각화!                             │
  │                                                         │
  │  ─────────────────────────────────────────────────────  │
  │  [ALGO] 대표적인 차원 축소 알고리즘                      │
  │  ─────────────────────────────────────────────────────  │
  │                                                         │
  │  1. t-SNE (t-distributed Stochastic Neighbor Embedding) │
  │     * 가까운 점들의 관계를 보존                         │
  │     * 클러스터 시각화에 최적                            │
  │     * 느림 (O(n²)), 대용량에 부적합                     │
  │     * 하이퍼파라미터: perplexity (5~50)                 │
  │                                                         │
  │  2. UMAP (Uniform Manifold Approximation & Projection)  │
  │     * t-SNE보다 빠름                                    │
  │     * 전역 구조도 어느 정도 보존                        │
  │     * 최근 더 많이 사용됨                               │
  │     * 하이퍼파라미터: n_neighbors, min_dist             │
  │                                                         │
  │  3. PCA (Principal Component Analysis)                  │
  │     * 가장 빠름, 선형 변환                              │
  │     * 분산을 최대화하는 축 선택                         │
  │     * 비선형 관계 포착 어려움                           │
  │                                                         │
  │  [TIP] 선택 가이드:                                     │
  │  * 클러스터 확인: t-SNE 또는 UMAP                       │
  │  * 빠른 탐색: PCA (선처리 후 t-SNE 적용도 가능)         │
  │  * 대용량 (10만+): UMAP 권장                            │
  └─────────────────────────────────────────────────────────┘
    """)
    
    if not os.getenv("OPENAI_API_KEY"):
        print("\n[!] OPENAI_API_KEY 환경변수를 설정해주세요!")
        return
    
    # t-SNE 시각화 실습 (실제 구현)
    print_section_header("t-SNE 시각화 실습", "[CODE]")
    
    # 시각화 라이브러리 확인
    try:
        from sklearn.manifold import TSNE
        tsne_available = True
    except ImportError:
        tsne_available = False
        print("\n[!] scikit-learn이 설치되지 않았습니다.")
        print("   설치: pip install scikit-learn")
    
    try:
        import matplotlib.pyplot as plt
        matplotlib_available = True
    except ImportError:
        matplotlib_available = False
        print("\n[!] matplotlib이 설치되지 않았습니다.")
        print("   설치: pip install matplotlib")
    
    # 샘플 텍스트 (카테고리별로 구분)
    texts_by_category = {
        "프로그래밍": [
            "Python is a great programming language",
            "JavaScript is used for web development",
            "Java is popular for enterprise applications",
            "C++ is used for system programming",
        ],
        "음식": [
            "Pizza is my favorite Italian food",
            "Sushi is a traditional Japanese dish",
            "Tacos are delicious Mexican food",
            "Pasta with tomato sauce is amazing",
        ],
        "스포츠": [
            "Soccer is the most popular sport worldwide",
            "Basketball requires great athleticism",
            "Tennis is an individual sport",
            "Swimming is excellent exercise",
        ],
    }
    
    # 텍스트와 라벨 준비
    all_texts = []
    labels = []
    for category, texts in texts_by_category.items():
        all_texts.extend(texts)
        labels.extend([category] * len(texts))
    
    print(f"\n총 {len(all_texts)}개 텍스트 (3개 카테고리)")
    for category, texts in texts_by_category.items():
        print(f"  * {category}: {len(texts)}개")
    
    # 임베딩 생성
    generator = EmbeddingGenerator()
    print("\n[...] 임베딩 생성 중...")
    embeddings = generator.get_embeddings_batch(all_texts)
    embeddings_array = np.array(embeddings)
    print(f"[OK] 임베딩 완료: {embeddings_array.shape}")
    
    if tsne_available and matplotlib_available:
        # t-SNE 실행
        print("\n[...] t-SNE 차원 축소 중...")
        tsne = TSNE(
            n_components=2,      # 2차원으로 축소
            perplexity=5,        # 작은 데이터셋이므로 낮은 값
            random_state=42,     # 재현성
            max_iter=1000        # 반복 횟수 (scikit-learn 1.5+에서 n_iter → max_iter)
        )
        embeddings_2d = tsne.fit_transform(embeddings_array)
        print(f"[OK] t-SNE 완료: {embeddings_2d.shape}")
        
        # 시각화 (텍스트 출력)
        print_section_header("시각화 결과 (텍스트)", "[CHART]")
        
        # 카테고리별 좌표 출력
        colors = {"프로그래밍": "🔵", "음식": "🟢", "스포츠": "🔴"}
        
        print("\n좌표 (x, y):")
        print(f"{'─'*60}")
        for i, (text, label) in enumerate(zip(all_texts, labels)):
            x, y = embeddings_2d[i]
            icon = colors.get(label, "⚪")
            print(f"  {icon} ({x:6.2f}, {y:6.2f}) {text[:40]}...")
        
        # ASCII 시각화
        print(f"\n{'─'*60}")
        print("ASCII 산점도 (대략적인 위치):")
        print(f"{'─'*60}")
        
        # 좌표 정규화 (0~40 범위)
        x_min, x_max = embeddings_2d[:, 0].min(), embeddings_2d[:, 0].max()
        y_min, y_max = embeddings_2d[:, 1].min(), embeddings_2d[:, 1].max()
        
        # 그리드 생성
        grid_width, grid_height = 60, 20
        grid = [[' ' for _ in range(grid_width)] for _ in range(grid_height)]
        
        # 점 배치
        category_symbols = {"프로그래밍": 'P', "음식": 'F', "스포츠": 'S'}
        for i, (label) in enumerate(labels):
            x_norm = int((embeddings_2d[i, 0] - x_min) / (x_max - x_min + 1e-10) * (grid_width - 1))
            y_norm = int((embeddings_2d[i, 1] - y_min) / (y_max - y_min + 1e-10) * (grid_height - 1))
            y_norm = grid_height - 1 - y_norm  # y축 반전
            grid[y_norm][x_norm] = category_symbols[label]
        
        # 그리드 출력
        print("  +" + "-" * grid_width + "+")
        for row in grid:
            print("  |" + "".join(row) + "|")
        print("  +" + "-" * grid_width + "+")
        print(f"  범례: P=프로그래밍, F=음식, S=스포츠")
        
        # matplotlib 차트 저장 (선택적)
        print_section_header("차트 파일 저장", "[FILE]")
        
        try:
            # 한글 폰트 설정 (Windows: Malgun Gothic, Mac: AppleGothic, Linux: NanumGothic)
            import platform
            system = platform.system()
            
            if system == "Windows":
                plt.rcParams['font.family'] = 'Malgun Gothic'
            elif system == "Darwin":  # macOS
                plt.rcParams['font.family'] = 'AppleGothic'
            else:  # Linux
                plt.rcParams['font.family'] = 'NanumGothic'
            
            # 마이너스 기호 깨짐 방지
            plt.rcParams['axes.unicode_minus'] = False
            
            plt.figure(figsize=(10, 8))
            
            color_map = {"프로그래밍": "blue", "음식": "green", "스포츠": "red"}
            
            for category in texts_by_category.keys():
                mask = [l == category for l in labels]
                indices = [i for i, m in enumerate(mask) if m]
                plt.scatter(
                    embeddings_2d[indices, 0],
                    embeddings_2d[indices, 1],
                    c=color_map[category],
                    label=category,
                    s=100,
                    alpha=0.7
                )
            
            # 텍스트 라벨 추가
            for i, text in enumerate(all_texts):
                plt.annotate(
                    text[:20] + "...",
                    (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                    fontsize=8,
                    alpha=0.7
                )
            
            plt.title("t-SNE 임베딩 시각화 (카테고리별 클러스터)")
            plt.xlabel("차원 1")
            plt.ylabel("차원 2")
            plt.legend()
            plt.tight_layout()
            
            # 파일 저장
            output_path = Path(__file__).parent / "embedding_tsne_demo.png"
            plt.savefig(output_path, dpi=150)
            plt.close()
            
            print(f"[OK] 차트 저장 완료: {output_path}")
            print("   → 이 파일을 열어 시각화를 확인하세요!")
            
        except Exception as e:
            print(f"[!] 차트 저장 실패: {e}")
            print("   (텍스트 시각화로 결과는 확인할 수 있습니다)")
    
    else:
        print("\n[!] 시각화 라이브러리가 없어 코드 예시만 제공합니다.")
        print("""
  [CODE] t-SNE 시각화 코드:
  ┌─────────────────────────────────────────────────────
  │ from sklearn.manifold import TSNE
  │ import matplotlib.pyplot as plt
  │ 
  │ # 1. 임베딩 준비 (N x 1536 배열)
  │ embeddings = np.array([...])
  │ 
  │ # 2. t-SNE 실행
  │ tsne = TSNE(n_components=2, perplexity=30, random_state=42)
  │ embeddings_2d = tsne.fit_transform(embeddings)
  │ 
  │ # 3. 시각화
  │ plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1])
  │ plt.title("Embedding Visualization")
  │ plt.savefig("embeddings.png")
  └─────────────────────────────────────────────────────
        """)
    
    # 시각화 해석 가이드
    print_section_header("시각화 해석 가이드", "[TIP]")
    print("""
  ┌─────────────────────────────────────────────────────────┐
  │  [!] 시각화 결과 해석 시 주의사항                        │
  │  ─────────────────────────────────────────────────────  │
  │                                                         │
  │  1. 클러스터 형성 확인                                   │
  │     * 같은 카테고리가 뭉쳐있으면 → 임베딩 품질 좋음     │
  │     * 섞여있으면 → 해당 개념이 모델에서 구분 안 됨      │
  │                                                         │
  │  2. 거리 해석 주의                                       │
  │     * t-SNE/UMAP의 거리는 절대적 의미 없음              │
  │     * "가까움/멂"만 의미 있음, 정확한 거리 아님         │
  │     * 여러 번 실행하면 모양이 달라질 수 있음 (랜덤성)   │
  │                                                         │
  │  3. 파라미터 영향                                        │
  │     * perplexity 높음 → 전역 구조 강조                  │
  │     * perplexity 낮음 → 지역 구조 강조                  │
  │     * 데이터 크기에 따라 조정 필요                      │
  │                                                         │
  │  4. 실무 활용                                            │
  │     * 데이터 품질 확인 (이상치 탐지)                    │
  │     * 클러스터링 결과 검증                              │
  │     * 새 카테고리 발견                                  │
  └─────────────────────────────────────────────────────────┘
    """)
    
    # 핵심 포인트
    print_key_points([
        "- 차원 축소: 1536차원 → 2D로 시각화하여 이해",
        "- t-SNE: 가까운 관계 보존, 클러스터 시각화에 최적",
        "- UMAP: t-SNE보다 빠름, 대용량에 적합",
        "- 해석 주의: 시각화 거리 ≠ 실제 유사도",
        "- 활용: 데이터 품질 확인, 클러스터 검증, 이상치 탐지"
    ], "임베딩 시각화 핵심 포인트")


# ============================================================================
# 7. 오픈소스 임베딩 모델
# ============================================================================

def demo_sentence_transformers():
    """실습 7: 오픈소스 임베딩 모델 - Sentence Transformers 소개"""
    print("\n" + "="*80)
    print("[7] 실습 7: 오픈소스 임베딩 모델 - Sentence Transformers 소개")
    print("="*80)
    print("목표: OpenAI 외 무료 오픈소스 임베딩 모델 이해")
    print("핵심: 비용 절감, 오프라인 사용, 커스터마이징 가능")
    
    # Sentence Transformers 소개
    print_section_header("Sentence Transformers란?", "[INFO]")
    print("""
  ┌─────────────────────────────────────────────────────────┐
  │  [LIB] Sentence Transformers (sentence-transformers)    │
  │  ─────────────────────────────────────────────────────  │
  │                                                         │
  │  * Hugging Face 기반 문장 임베딩 라이브러리             │
  │  * 수백 개의 사전 훈련된 모델 제공                      │
  │  * MIT 라이선스 (상업적 사용 가능)                      │
  │  * 로컬 실행 → API 비용 없음!                          │
  │                                                         │
  │  [설치]                                                 │
  │  pip install sentence-transformers                      │
  │                                                         │
  │  [기본 사용법]                                          │
  │  ┌─────────────────────────────────────────────────    │
  │  │ from sentence_transformers import SentenceTransformer│
  │  │                                                      │
  │  │ # 모델 로드 (최초 실행 시 다운로드)                  │
  │  │ model = SentenceTransformer('all-MiniLM-L6-v2')     │
  │  │                                                      │
  │  │ # 임베딩 생성                                        │
  │  │ sentences = ["Hello world", "How are you?"]         │
  │  │ embeddings = model.encode(sentences)                 │
  │  │ # embeddings.shape: (2, 384)                        │
  │  └─────────────────────────────────────────────────    │
  └─────────────────────────────────────────────────────────┘
    """)
    
    # 인기 모델 비교
    print_section_header("인기 오픈소스 임베딩 모델", "[LIST]")
    print("""
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │  [CMP] 주요 오픈소스 임베딩 모델 비교                                        │
  │  ─────────────────────────────────────────────────────────────────────────  │
  │                                                                             │
  │  모델명                      │ 차원  │ 크기   │ 속도   │ 품질  │ 특징      │
  │  ──────────────────────────┼──────┼───────┼───────┼──────┼───────────│
  │  all-MiniLM-L6-v2          │ 384  │ 80MB  │ 빠름  │ 좋음  │ 가장 인기  │
  │  all-mpnet-base-v2         │ 768  │ 420MB │ 중간  │ 높음  │ 균형잡힘  │
  │  paraphrase-multilingual-  │ 768  │ 1GB   │ 느림  │ 높음  │ 다국어    │
  │    MiniLM-L12-v2           │      │       │       │      │ (한글 OK) │
  │  e5-large-v2               │ 1024 │ 1.3GB │ 느림  │ 최고  │ SOTA급    │
  │  bge-large-en-v1.5         │ 1024 │ 1.3GB │ 느림  │ 최고  │ 중국 BAAI │
  │  ──────────────────────────┴──────┴───────┴───────┴──────┴───────────│
  │                                                                             │
  │  [TIP] 선택 가이드:                                                         │
  │  * 빠른 프로토타입: all-MiniLM-L6-v2 (작고 빠름)                            │
  │  * 프로덕션 품질: all-mpnet-base-v2 또는 e5-large                           │
  │  * 한글 지원: paraphrase-multilingual-MiniLM-L12-v2                         │
  │  * 최고 성능: bge-large-en-v1.5 또는 e5-large-v2                            │
  └─────────────────────────────────────────────────────────────────────────────┘
    """)
    
    # OpenAI vs Sentence Transformers 비교
    print_section_header("OpenAI vs Sentence Transformers", "[vs]")
    print("""
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  [CMP] 비교표                                                            │
  │  ─────────────────────────────────────────────────────────────────────  │
  │                                                                         │
  │  항목            │ OpenAI                 │ Sentence Transformers      │
  │  ───────────────┼───────────────────────┼───────────────────────────  │
  │  비용           │ $0.02 / 1M 토큰        │ 무료 (로컬 실행)            │
  │  속도           │ 네트워크 지연 있음      │ GPU 있으면 매우 빠름       │
  │  품질           │ 최상급                 │ 모델에 따라 다름           │
  │  오프라인       │ ✗ 불가                 │ ✓ 가능                     │
  │  커스터마이징   │ ✗ 불가 (API만 제공)    │ ✓ 파인튜닝 가능            │
  │  다국어         │ ✓ 우수                 │ 모델에 따라 다름           │
  │  설치           │ pip install openai     │ pip install sentence-      │
  │                 │                        │   transformers             │
  │  ───────────────┴───────────────────────┴───────────────────────────  │
  │                                                                         │
  │  [TIP] 언제 무엇을 선택할까?                                            │
  │  ─────────────────────────────────────────────────────────────────────  │
  │                                                                         │
  │  OpenAI 선택:                                                           │
  │  * 최고 품질이 필요할 때                                                │
  │  * 다국어 지원이 중요할 때                                              │
  │  * 인프라 관리 없이 빠르게 시작할 때                                    │
  │  * 사용량이 적을 때 (비용 부담 적음)                                    │
  │                                                                         │
  │  Sentence Transformers 선택:                                            │
  │  * 비용 절감이 중요할 때 (대용량 처리)                                  │
  │  * 오프라인/에어갭 환경                                                 │
  │  * 데이터 보안 (외부 전송 불가)                                         │
  │  * 커스텀 도메인 파인튜닝 필요                                          │
  │  * GPU 서버가 있을 때                                                   │
  └─────────────────────────────────────────────────────────────────────────┘
    """)
    
    # 실제 사용 예시
    print_section_header("Sentence Transformers 사용 예시", "[CODE]")
    
    # 라이브러리 확인
    try:
        from sentence_transformers import SentenceTransformer
        st_available = True
    except ImportError:
        st_available = False
    
    if st_available:
        print("\n[OK] sentence-transformers가 설치되어 있습니다.")
        print("\n[...] 모델 로딩 중 (최초 실행 시 다운로드)...")
        
        try:
            # 가벼운 모델 사용
            model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # 테스트 문장
            sentences = [
                "Python is a programming language",
                "Java is also a programming language",
                "I love eating pizza",
            ]
            
            print(f"[OK] 모델 로드 완료: all-MiniLM-L6-v2")
            print(f"\n테스트 문장:")
            for i, s in enumerate(sentences, 1):
                print(f"  {i}. {s}")
            
            # 임베딩 생성
            embeddings = model.encode(sentences)
            
            print(f"\n임베딩 결과:")
            print(f"  * Shape: {embeddings.shape}")
            print(f"  * 차원: {embeddings.shape[1]}")
            
            # 유사도 계산
            print(f"\n코사인 유사도 비교:")
            sim_1_2 = cosine_similarity(embeddings[0].tolist(), embeddings[1].tolist())
            sim_1_3 = cosine_similarity(embeddings[0].tolist(), embeddings[2].tolist())
            
            print(f"  Python vs Java: {sim_1_2:.4f} (프로그래밍 언어끼리)")
            print(f"  Python vs Pizza: {sim_1_3:.4f} (다른 주제)")
            print(f"\n  → 같은 주제는 유사도가 높음!")
            
        except Exception as e:
            print(f"\n[!] 실행 중 오류: {e}")
            print("   모델 다운로드에 인터넷 연결이 필요합니다.")
    
    else:
        print("\n[!] sentence-transformers가 설치되지 않았습니다.")
        print("   설치: pip install sentence-transformers")
        print("""
  [CODE] 설치 후 사용 예시:
  ┌─────────────────────────────────────────────────────
  │ from sentence_transformers import SentenceTransformer
  │ 
  │ # 모델 로드
  │ model = SentenceTransformer('all-MiniLM-L6-v2')
  │ 
  │ # 임베딩 생성
  │ sentences = ["Hello world", "안녕하세요"]
  │ embeddings = model.encode(sentences)
  │ 
  │ print(embeddings.shape)  # (2, 384)
  └─────────────────────────────────────────────────────
        """)
    
    # 파인튜닝 가이드
    print_section_header("파인튜닝 가이드 (심화)", "[ADV]")
    print("""
  ┌─────────────────────────────────────────────────────────┐
  │  [!] 도메인 특화 파인튜닝                                │
  │  ─────────────────────────────────────────────────────  │
  │                                                         │
  │  기본 모델이 특정 도메인에서 성능이 낮을 때:            │
  │  * 법률/의료/금융 전문 용어                             │
  │  * 회사 내부 문서 스타일                                │
  │  * 특정 언어/방언                                       │
  │                                                         │
  │  [방법]                                                 │
  │  1. 대조 학습 (Contrastive Learning)                   │
  │     - 유사한 문장 쌍 / 다른 문장 쌍 데이터 준비         │
  │     - 유사한 것은 가깝게, 다른 것은 멀게 학습           │
  │                                                         │
  │  2. 필요 데이터                                         │
  │     - 최소 1,000~10,000개 문장 쌍                       │
  │     - (query, positive, negative) 형태                  │
  │                                                         │
  │  [CODE] 파인튜닝 예시:                                  │
  │  ┌─────────────────────────────────────────────────    │
  │  │ from sentence_transformers import (                  │
  │  │     SentenceTransformer, InputExample, losses       │
  │  │ )                                                    │
  │  │ from torch.utils.data import DataLoader              │
  │  │                                                      │
  │  │ model = SentenceTransformer('all-MiniLM-L6-v2')     │
  │  │                                                      │
  │  │ # 학습 데이터 준비                                   │
  │  │ train_examples = [                                   │
  │  │     InputExample(texts=["질문", "정답 문서"], label=1.0),│
  │  │     InputExample(texts=["질문", "무관 문서"], label=0.0),│
  │  │ ]                                                    │
  │  │                                                      │
  │  │ train_dataloader = DataLoader(train_examples, batch_size=16)│
  │  │ train_loss = losses.CosineSimilarityLoss(model)     │
  │  │                                                      │
  │  │ model.fit(                                           │
  │  │     train_objectives=[(train_dataloader, train_loss)],│
  │  │     epochs=3                                         │
  │  │ )                                                    │
  │  └─────────────────────────────────────────────────    │
  │                                                         │
  │  [TIP] 파인튜닝 시기:                                   │
  │  * 기본 모델로 Recall@5 < 80% 일 때                     │
  │  * 도메인 용어가 많아서 검색 품질 낮을 때               │
  │  * 데이터가 충분할 때 (최소 1,000쌍)                    │
  └─────────────────────────────────────────────────────────┘
    """)
    
    # 핵심 포인트
    print_key_points([
        "- Sentence Transformers: 무료 오픈소스 임베딩 라이브러리",
        "- all-MiniLM-L6-v2: 가볍고 빠름, 프로토타입에 적합",
        "- OpenAI vs 오픈소스: 품질 vs 비용/커스터마이징",
        "- 파인튜닝: 도메인 특화 시 성능 향상 가능",
        "- 다국어: paraphrase-multilingual 모델 사용"
    ], "Sentence Transformers 핵심 포인트")


# ============================================================================
# 8. 임베딩 모델 비교
# ============================================================================

def demo_embedding_model_comparison():
    """실습 8: 임베딩 모델 비교 - small vs large 성능/비용 분석"""
    print("\n" + "="*80)
    print("[8] 실습 8: 임베딩 모델 비교 - small vs large 성능/비용 분석")
    print("="*80)
    print("목표: 모델 선택 시 고려해야 할 요소 이해")
    print("핵심: 품질, 비용, 속도의 Trade-off")
    
    # OpenAI 임베딩 모델 비교
    print_section_header("OpenAI 임베딩 모델 비교", "[CMP]")
    print("""
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │  [OpenAI] 임베딩 모델 상세 비교                                              │
  │  ─────────────────────────────────────────────────────────────────────────  │
  │                                                                             │
  │  모델                     │ 차원  │ 가격(/1M토큰) │ 최대 토큰 │ 특징       │
  │  ────────────────────────┼──────┼──────────────┼─────────┼────────────│
  │  text-embedding-3-small  │ 1536 │ $0.02        │ 8,191   │ 가성비 최고│
  │  text-embedding-3-large  │ 3072 │ $0.13        │ 8,191   │ 최고 품질  │
  │  text-embedding-ada-002  │ 1536 │ $0.10        │ 8,191   │ 레거시    │
  │  ────────────────────────┴──────┴──────────────┴─────────┴────────────│
  │                                                                             │
  │  [!] text-embedding-3 시리즈 특징:                                          │
  │  * 차원 축소 지원: dimensions 파라미터로 256~3072 지정 가능                 │
  │  * 예: small 모델을 256차원으로 사용 → 저장 공간 절약                       │
  │                                                                             │
  │  [CODE] 차원 축소 예시:                                                     │
  │  response = client.embeddings.create(                                       │
  │      model="text-embedding-3-small",                                        │
  │      input="Hello world",                                                   │
  │      dimensions=256  # 1536 대신 256차원                                    │
  │  )                                                                          │
  └─────────────────────────────────────────────────────────────────────────────┘
    """)
    
    # 비용 계산 예시
    print_section_header("비용 계산 예시", "[CALC]")
    print("""
  ┌─────────────────────────────────────────────────────────┐
  │  [CASE] 월간 비용 시뮬레이션                            │
  │  ─────────────────────────────────────────────────────  │
  │                                                         │
  │  시나리오: RAG 시스템 운영                               │
  │  * 문서 10,000개 인덱싱 (각 500 토큰)                   │
  │  * 일일 쿼리 1,000개 (각 50 토큰)                       │
  │                                                         │
  │  ─────────────────────────────────────────────────────  │
  │  1. 초기 인덱싱 비용 (1회성)                            │
  │  ─────────────────────────────────────────────────────  │
  │  * 총 토큰: 10,000 × 500 = 5M 토큰                      │
  │                                                         │
  │  │ 모델                     │ 비용        │             │
  │  │ ────────────────────────┼────────────│             │
  │  │ text-embedding-3-small  │ $0.10      │             │
  │  │ text-embedding-3-large  │ $0.65      │             │
  │  │ text-embedding-ada-002  │ $0.50      │             │
  │                                                         │
  │  ─────────────────────────────────────────────────────  │
  │  2. 월간 쿼리 비용                                      │
  │  ─────────────────────────────────────────────────────  │
  │  * 월간 토큰: 1,000 × 50 × 30 = 1.5M 토큰              │
  │                                                         │
  │  │ 모델                     │ 월 비용     │ 연 비용    │
  │  │ ────────────────────────┼────────────┼───────────│
  │  │ text-embedding-3-small  │ $0.03      │ $0.36     │
  │  │ text-embedding-3-large  │ $0.20      │ $2.34     │
  │                                                         │
  │  [결론]                                                 │
  │  * Small로도 대부분의 용도에 충분!                      │
  │  * Large는 법률/의료 등 정밀도가 중요할 때만            │
  │  * 차이: 연간 $2 정도 (소규모 기준)                    │
  └─────────────────────────────────────────────────────────┘
    """)
    
    # 품질 비교 (벤치마크)
    print_section_header("품질 비교 (MTEB 벤치마크)", "[BENCH]")
    print("""
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  [MTEB] Massive Text Embedding Benchmark 결과                           │
  │  ─────────────────────────────────────────────────────────────────────  │
  │                                                                         │
  │  MTEB는 56개 데이터셋에서 임베딩 모델 성능을 평가하는 표준 벤치마크      │
  │                                                                         │
  │  모델                      │ 평균 점수 │ 검색 점수 │ 클러스터링 │ 분류  │
  │  ─────────────────────────┼──────────┼──────────┼───────────┼──────│
  │  text-embedding-3-large   │ 64.6     │ 55.4     │ 49.0      │ 75.5 │
  │  text-embedding-3-small   │ 62.3     │ 51.8     │ 44.0      │ 73.5 │
  │  text-embedding-ada-002   │ 61.0     │ 49.2     │ 45.9      │ 70.9 │
  │  ─────────────────────────┼──────────┼──────────┼───────────┼──────│
  │  bge-large-en-v1.5        │ 64.2     │ 54.3     │ 46.1      │ 75.0 │
  │  e5-large-v2              │ 62.2     │ 50.6     │ 44.5      │ 73.8 │
  │  all-mpnet-base-v2        │ 57.8     │ 43.8     │ 43.7      │ 65.0 │
  │  ─────────────────────────┴──────────┴──────────┴───────────┴──────│
  │                                                                         │
  │  [해석]                                                                 │
  │  * OpenAI large: 최고 수준, 특히 검색 태스크에서 강함                   │
  │  * OpenAI small: large 대비 2~3점 낮음, 비용 대비 효율적                │
  │  * bge/e5: OpenAI와 비슷한 수준, 무료로 사용 가능                       │
  │  * mpnet: 가볍지만 품질 차이 있음                                       │
  │                                                                         │
  │  [!] 실무 의미:                                                         │
  │  * 점수 2~3점 차이 = Recall@5에서 약 1~2% 차이                          │
  │  * 대부분의 RAG 시스템에서는 체감 어려움                                │
  │  * 법률/의료처럼 1%도 중요한 도메인에서만 large 권장                    │
  └─────────────────────────────────────────────────────────────────────────┘
    """)
    
    # 실제 비교 실험 (선택적)
    if os.getenv("OPENAI_API_KEY"):
        print_section_header("실제 비교 실험", "[EXP]")
        print("\n[...] small vs large 모델 비교 중...")
        
        try:
            from openai import OpenAI
            client = get_openai_client()
            
            test_texts = [
                "What is machine learning?",
                "Machine learning is a subset of artificial intelligence",
                "I love eating pizza for dinner",
            ]
            
            # Small 모델
            response_small = client.embeddings.create(
                model="text-embedding-3-small",
                input=test_texts
            )
            embeddings_small = [d.embedding for d in response_small.data]
            
            # Large 모델
            response_large = client.embeddings.create(
                model="text-embedding-3-large",
                input=test_texts
            )
            embeddings_large = [d.embedding for d in response_large.data]
            
            print(f"\n텍스트:")
            for i, t in enumerate(test_texts, 1):
                print(f"  {i}. {t}")
            
            print(f"\n유사도 비교 (텍스트 1 vs 나머지):")
            print(f"{'─'*60}")
            print(f"{'비교 대상':<30} {'Small':<12} {'Large':<12}")
            print(f"{'─'*60}")
            
            # Small 유사도
            sim_small_1_2 = cosine_similarity(embeddings_small[0], embeddings_small[1])
            sim_small_1_3 = cosine_similarity(embeddings_small[0], embeddings_small[2])
            
            # Large 유사도
            sim_large_1_2 = cosine_similarity(embeddings_large[0], embeddings_large[1])
            sim_large_1_3 = cosine_similarity(embeddings_large[0], embeddings_large[2])
            
            print(f"{'vs ML 설명 (관련)':<30} {sim_small_1_2:<12.4f} {sim_large_1_2:<12.4f}")
            print(f"{'vs Pizza (무관)':<30} {sim_small_1_3:<12.4f} {sim_large_1_3:<12.4f}")
            print(f"{'─'*60}")
            
            # 차이 분석
            gap_small = sim_small_1_2 - sim_small_1_3
            gap_large = sim_large_1_2 - sim_large_1_3
            
            print(f"\n관련/무관 점수 차이:")
            print(f"  Small: {gap_small:.4f}")
            print(f"  Large: {gap_large:.4f}")
            
            if gap_large > gap_small:
                print(f"\n  → Large 모델이 관련/무관을 더 잘 구분함!")
            else:
                print(f"\n  → 이 예시에서는 차이가 미미함")
            
            print(f"\n차원 비교:")
            print(f"  Small: {len(embeddings_small[0])} 차원")
            print(f"  Large: {len(embeddings_large[0])} 차원")
            
        except Exception as e:
            print(f"\n[!] 비교 실험 실패: {e}")
    
    # 모델 선택 가이드
    print_section_header("모델 선택 가이드", "[GUIDE]")
    print("""
  ┌─────────────────────────────────────────────────────────┐
  │  [FLOW] 모델 선택 의사결정 트리                         │
  │  ─────────────────────────────────────────────────────  │
  │                                                         │
  │  Q1. 비용이 가장 중요한가?                              │
  │   │                                                     │
  │   ├─ YES → Q2. GPU 서버가 있는가?                       │
  │   │         │                                           │
  │   │         ├─ YES → Sentence Transformers (무료)       │
  │   │         │        (all-MiniLM-L6-v2 또는 bge-large)  │
  │   │         │                                           │
  │   │         └─ NO → text-embedding-3-small              │
  │   │                 (가장 저렴한 API)                   │
  │   │                                                     │
  │   └─ NO → Q3. 최고 품질이 필요한가?                     │
  │            │                                            │
  │            ├─ YES → text-embedding-3-large              │
  │            │        (법률/의료/금융)                    │
  │            │                                            │
  │            └─ NO → text-embedding-3-small               │
  │                    (대부분의 RAG 시스템)                │
  │                                                         │
  │  ─────────────────────────────────────────────────────  │
  │  [TIP] 추가 고려사항                                    │
  │  ─────────────────────────────────────────────────────  │
  │                                                         │
  │  * 다국어 (한글): OpenAI > 오픈소스 (multilingual 제외) │
  │  * 오프라인 필수: Sentence Transformers only            │
  │  * 데이터 보안: 로컬 모델 (외부 전송 불가)              │
  │  * 파인튜닝 필요: Sentence Transformers (OpenAI 불가)   │
  │  * 빠른 PoC: OpenAI (설치 없이 바로 사용)               │
  └─────────────────────────────────────────────────────────┘
    """)
    
    # 핵심 포인트
    print_key_points([
        "- text-embedding-3-small: 가성비 최고, 대부분의 용도에 충분",
        "- text-embedding-3-large: 최고 품질, 정밀도가 중요한 도메인용",
        "- 비용 차이: small $0.02 vs large $0.13 (6.5배 차이)",
        "- 품질 차이: MTEB 기준 약 2~3점 (실무에서 체감 어려움)",
        "- 오픈소스: bge-large, e5-large가 OpenAI급 성능"
    ], "임베딩 모델 비교 핵심 포인트")


# ============================================================================
# 9. 한글-영어 임베딩 비교
# ============================================================================

def demo_korean_english_comparison():
    """실습 9: 한글-영어 임베딩 비교 - 다국어 의미 정렬(Alignment) 실험"""
    print("\n" + "="*80)
    print("[9] 실습 9: 한글-영어 임베딩 비교 - 다국어 의미 정렬 실험")
    print("="*80)
    print("목표: 한글과 영어가 같은 의미일 때 임베딩이 얼마나 유사한지 확인")
    print("핵심: 다국어 임베딩 모델의 Cross-lingual Alignment 품질 비교")
    
    # 다국어 정렬이란?
    print_section_header("다국어 의미 정렬(Cross-lingual Alignment)이란?", "[INFO]")
    print("""
  ┌─────────────────────────────────────────────────────────┐
  │  [!] 문제: 한글 질문으로 영어 문서를 검색할 수 있을까?   │
  │  ─────────────────────────────────────────────────────  │
  │                                                         │
  │  이상적인 다국어 임베딩:                                 │
  │  • "파이썬은 프로그래밍 언어다"                         │
  │  • "Python is a programming language"                  │
  │  → 두 벡터가 벡터 공간에서 가까워야 함!                │
  │                                                         │
  │  ─────────────────────────────────────────────────────  │
  │  [DEMO] 벡터 공간 시각화                                │
  │  ─────────────────────────────────────────────────────  │
  │                                                         │
  │         영어                한글                        │
  │  ┌─────────────────────────────────────────────┐       │
  │  │         "Python"    "파이썬"                 │       │
  │  │              *  ←──→  *   ← 가까움 (좋음)   │       │
  │  │                                              │       │
  │  │         "pizza"     "피자"                  │       │
  │  │              *  ←──→  *   ← 가까움 (좋음)   │       │
  │  │                                              │       │
  │  │  "Python" *                                  │       │
  │  │                         * "피자"            │       │
  │  │              ↑                              │       │
  │  │           멂 (다른 주제)                    │       │
  │  └─────────────────────────────────────────────┘       │
  │                                                         │
  │  [TIP] 이것이 왜 중요한가?                              │
  │  • 한글 RAG에서 영어 기술 문서 검색 가능               │
  │  • 다국어 고객센터 챗봇 구현 가능                       │
  │  • 번역 없이 교차 언어 검색 가능                        │
  └─────────────────────────────────────────────────────────┘
    """)
    
    if not os.getenv("OPENAI_API_KEY"):
        print("\n[!] OPENAI_API_KEY 환경변수를 설정해주세요!")
        return
    
    # 테스트 문장 준비 (한글-영어 쌍)
    test_pairs = [
        {
            "korean": "파이썬은 프로그래밍 언어입니다",
            "english": "Python is a programming language",
            "category": "프로그래밍"
        },
        {
            "korean": "나는 피자를 좋아합니다",
            "english": "I love eating pizza",
            "category": "음식"
        },
        {
            "korean": "머신러닝은 인공지능의 한 분야입니다",
            "english": "Machine learning is a subset of artificial intelligence",
            "category": "AI/ML"
        },
        {
            "korean": "오늘 날씨가 좋습니다",
            "english": "The weather is nice today",
            "category": "날씨"
        },
    ]
    
    print_section_header("테스트 문장 쌍", "[DATA]")
    print(f"\n{'─'*70}")
    print(f"{'카테고리':<12} {'한글':<30} {'영어':<30}")
    print(f"{'─'*70}")
    for pair in test_pairs:
        print(f"{pair['category']:<12} {pair['korean']:<30} {pair['english']:<30}")
    print(f"{'─'*70}")
    
    # ========== OpenAI 임베딩 테스트 ==========
    print_section_header("1. OpenAI 임베딩 다국어 정렬 테스트", "[OPENAI]")
    
    generator = EmbeddingGenerator()
    
    # 모든 텍스트 임베딩 생성
    all_texts = []
    labels = []
    for pair in test_pairs:
        all_texts.extend([pair['korean'], pair['english']])
        labels.extend([f"{pair['category']}_KO", f"{pair['category']}_EN"])
    
    print("\n[...] OpenAI 임베딩 생성 중...")
    openai_embeddings = generator.get_embeddings_batch(all_texts)
    print(f"[OK] {len(openai_embeddings)}개 임베딩 생성 완료")
    
    # 유사도 분석
    print(f"\n[분석 결과] 한글-영어 쌍별 코사인 유사도:")
    print(f"{'─'*70}")
    print(f"{'카테고리':<12} {'한글 vs 영어 (같은 의미)':<25} {'해석':<20}")
    print(f"{'─'*70}")
    
    openai_same_meaning_sims = []
    for i, pair in enumerate(test_pairs):
        ko_emb = openai_embeddings[i * 2]
        en_emb = openai_embeddings[i * 2 + 1]
        sim = cosine_similarity(ko_emb, en_emb)
        openai_same_meaning_sims.append(sim)
        
        interpretation = interpret_cosine_similarity(sim)
        bar = visualize_similarity_bar(sim, 20)
        print(f"{pair['category']:<12} {bar} {sim:.4f}  {interpretation}")
    
    avg_openai_same = np.mean(openai_same_meaning_sims)
    print(f"{'─'*70}")
    print(f"{'평균':<12} {'':<21} {avg_openai_same:.4f}")
    
    # 다른 의미 비교 (프로그래밍 한글 vs 음식 영어)
    print(f"\n[대조군] 다른 의미 쌍 비교:")
    print(f"{'─'*70}")
    
    # 프로그래밍 한글 vs 음식 영어
    ko_prog_emb = openai_embeddings[0]  # 파이썬 한글
    en_food_emb = openai_embeddings[3]  # pizza 영어
    sim_diff = cosine_similarity(ko_prog_emb, en_food_emb)
    bar_diff = visualize_similarity_bar(sim_diff, 20)
    print(f"'파이썬은 프로그래밍...' vs 'I love eating pizza'")
    print(f"  → {bar_diff} {sim_diff:.4f} (다른 주제)")
    
    print(f"\n[결론] OpenAI 임베딩:")
    print(f"  • 같은 의미 (한글-영어): 평균 {avg_openai_same:.4f}")
    print(f"  • 다른 의미 (한글-영어): {sim_diff:.4f}")
    print(f"  • 차이: {avg_openai_same - sim_diff:.4f}")
    
    if avg_openai_same > 0.8:
        print(f"  → [v] 뛰어난 다국어 정렬! 한글 RAG에 적합")
    elif avg_openai_same > 0.6:
        print(f"  → [~] 양호한 다국어 정렬. 대부분의 용도에 OK")
    else:
        print(f"  → [x] 다국어 정렬이 약함. multilingual 모델 권장")
    
    # ========== Sentence Transformers 비교 ==========
    print_section_header("2. Sentence Transformers 다국어 모델 비교", "[ST]")
    
    try:
        from sentence_transformers import SentenceTransformer
        st_available = True
    except ImportError:
        st_available = False
        print("\n[!] sentence-transformers가 설치되지 않았습니다.")
        print("   설치: pip install sentence-transformers")
    
    if st_available:
        # 다국어 모델 테스트
        multilingual_models = [
            ("paraphrase-multilingual-MiniLM-L12-v2", "다국어 특화"),
            ("all-MiniLM-L6-v2", "영어 중심 (비교용)"),
        ]
        
        results = {}
        
        for model_name, description in multilingual_models:
            print(f"\n[...] '{model_name}' 모델 로딩 중...")
            
            try:
                model = SentenceTransformer(model_name)
                
                # 임베딩 생성
                st_embeddings = model.encode(all_texts)
                
                # 같은 의미 유사도 계산
                same_meaning_sims = []
                for i, pair in enumerate(test_pairs):
                    ko_emb = st_embeddings[i * 2]
                    en_emb = st_embeddings[i * 2 + 1]
                    sim = cosine_similarity(ko_emb.tolist(), en_emb.tolist())
                    same_meaning_sims.append(sim)
                
                avg_sim = np.mean(same_meaning_sims)
                
                # 다른 의미 유사도
                sim_diff_st = cosine_similarity(
                    st_embeddings[0].tolist(),  # 파이썬 한글
                    st_embeddings[3].tolist()   # pizza 영어
                )
                
                results[model_name] = {
                    "avg_same": avg_sim,
                    "diff": sim_diff_st,
                    "gap": avg_sim - sim_diff_st,
                    "description": description
                }
                
                print(f"[OK] '{model_name}' 완료")
                print(f"     같은 의미 평균: {avg_sim:.4f}, 다른 의미: {sim_diff_st:.4f}, 차이: {avg_sim - sim_diff_st:.4f}")
                
            except Exception as e:
                print(f"[!] '{model_name}' 로드 실패: {e}")
        
        # 결과 비교표
        print_section_header("3. 모델별 다국어 정렬 비교", "[CMP]")
        print(f"\n{'─'*75}")
        print(f"{'모델':<40} {'같은 의미':<12} {'다른 의미':<12} {'차이(Gap)':<10}")
        print(f"{'─'*75}")
        
        # OpenAI 결과 추가
        print(f"{'OpenAI text-embedding-3-small':<40} {avg_openai_same:<12.4f} {sim_diff:<12.4f} {avg_openai_same - sim_diff:<10.4f}")
        
        for model_name, data in results.items():
            print(f"{model_name:<40} {data['avg_same']:<12.4f} {data['diff']:<12.4f} {data['gap']:<10.4f}")
        
        print(f"{'─'*75}")
        
        # 승자 결정
        all_results = {"OpenAI": {"gap": avg_openai_same - sim_diff, "avg_same": avg_openai_same}}
        all_results.update({k: {"gap": v["gap"], "avg_same": v["avg_same"]} for k, v in results.items()})
        
        best_model = max(all_results.items(), key=lambda x: x[1]["gap"])
        
        print(f"\n[*] 다국어 정렬 최고 모델: {best_model[0]}")
        print(f"    (같은 의미 유사도와 다른 의미 유사도 차이가 가장 큼)")
    
    # 실무 가이드
    print_section_header("한글 RAG 실무 가이드", "[GUIDE]")
    print("""
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  [CASE] 한글 RAG 시나리오별 권장 모델                                    │
  │  ─────────────────────────────────────────────────────────────────────  │
  │                                                                         │
  │  시나리오 1: 한글 문서 + 한글 질문                                       │
  │  ─────────────────────────────────────────────────────────────────────  │
  │  • OpenAI text-embedding-3-small (권장)                                │
  │  • 이유: 한글 품질 최상급, API로 간편 사용                              │
  │                                                                         │
  │  시나리오 2: 영어 문서 + 한글 질문 (또는 반대)                          │
  │  ─────────────────────────────────────────────────────────────────────  │
  │  • OpenAI (가장 안전한 선택)                                            │
  │  • paraphrase-multilingual-MiniLM-L12-v2 (비용 절감 시)                 │
  │  • 이유: 다국어 정렬(Alignment)이 중요                                  │
  │                                                                         │
  │  시나리오 3: 한글 전용, 비용 최소화                                     │
  │  ─────────────────────────────────────────────────────────────────────  │
  │  • paraphrase-multilingual-MiniLM-L12-v2 (무료)                         │
  │  • 또는 KoSimCSE (한글 특화 모델)                                       │
  │  • 이유: 로컬 실행, API 비용 없음                                       │
  │                                                                         │
  │  [!] 주의사항                                                           │
  │  ─────────────────────────────────────────────────────────────────────  │
  │  • 영어 중심 모델 (all-MiniLM 등)은 한글에서 성능 저하                  │
  │  • 반드시 한글 테스트 후 모델 선택!                                     │
  │  • 도메인 용어가 많으면 파인튜닝 고려                                   │
  └─────────────────────────────────────────────────────────────────────────┘
    """)
    
    # 핵심 포인트
    print_key_points([
        "- 다국어 정렬: 같은 의미의 한글/영어가 벡터 공간에서 가까운 정도",
        "- OpenAI: 다국어 정렬 우수, 한글 RAG에 안전한 선택",
        "- multilingual 모델: 명시적으로 다국어 학습된 모델 사용",
        "- 영어 중심 모델: all-MiniLM 등은 한글에서 성능 저하 가능",
        "- 실무 팁: 한글 테스트 데이터로 반드시 검증 후 모델 선택!"
    ], "한글-영어 임베딩 비교 핵심 포인트")


# ============================================================================
# 메인 실행
# ============================================================================

def parse_args():
    """명령줄 인자 파싱"""
    parser = argparse.ArgumentParser(
        description="NLP 기초 실습",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
실행 모드:
  python nlp_basics.py          # 전체 실습 (기본)
  python nlp_basics.py --demo   # 출력 위주 데모 (API 호출 최소화)
  python nlp_basics.py --run    # 실제 계산 + 시각화 파일 저장
  python nlp_basics.py --quick  # 핵심 실습만 (1~5번)

예시:
  python nlp_basics.py --run    # 모든 실습 실행 + PNG 파일 저장
  python nlp_basics.py --quick  # 빠른 데모 (기초만)
        """
    )
    
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--demo", 
        action="store_true",
        help="출력 위주 데모 모드 (API 호출 최소화, 빠른 실행)"
    )
    mode_group.add_argument(
        "--run", 
        action="store_true",
        help="실제 계산 모드 (모든 실험 실행 + 시각화 파일 저장)"
    )
    mode_group.add_argument(
        "--quick", 
        action="store_true",
        help="핵심 실습만 (1~5번, API 키 없어도 일부 가능)"
    )
    
    parser.add_argument(
        "--save-plots",
        action="store_true",
        default=True,
        help="시각화 결과를 PNG 파일로 저장 (기본: True)"
    )
    
    return parser.parse_args()


def main():
    """모든 데모 실행"""
    args = parse_args()
    
    # 실행 모드 결정
    if args.demo:
        mode = "demo"
        mode_desc = "데모 모드 (출력 위주)"
    elif args.run:
        mode = "run"
        mode_desc = "실행 모드 (전체 계산 + 파일 저장)"
    elif args.quick:
        mode = "quick"
        mode_desc = "퀵 모드 (핵심 실습 1~5번만)"
    else:
        mode = "full"
        mode_desc = "전체 실습"
    
    print("\n" + "="*80)
    print("[LAB] NLP 기초 실습")
    print(f"[MODE] {mode_desc}")
    print("="*80)
    
    print("\n[LIST] 실습 항목:")
    print("  1. tiktoken으로 토큰 이해하기 - GPT가 텍스트를 어떻게 보는가")
    print("  2. NLTK 전처리 파이프라인 - 토큰화, 불용어, 표제어 추출")
    print("  3. OpenAI 임베딩 생성 - 텍스트를 벡터로 변환")
    print("  4. 코사인 유사도 계산 - 벡터 간 유사성 측정")
    print("  5. 간단한 검색 엔진 - 의미 기반 문장 검색")
    
    if mode != "quick":
        print("  6. 임베딩 시각화 - t-SNE로 벡터 공간 이해하기")
        print("  7. 오픈소스 임베딩 모델 - Sentence Transformers 소개")
        print("  8. 임베딩 모델 비교 - small vs large 성능/비용 분석")
        print("  9. 한글-영어 임베딩 비교 - 다국어 의미 정렬 실험 🆕")
    
    if mode == "demo":
        print("\n[INFO] 데모 모드: API 호출을 최소화하여 빠르게 실행합니다.")
    elif mode == "run":
        print("\n[INFO] 실행 모드: 모든 실험을 실제로 실행하고 결과를 저장합니다.")
        print(f"       시각화 파일 저장: {args.save_plots}")
    
    # NLTK 데이터 다운로드
    download_nltk_data()
    
    try:
        # 1. tiktoken 데모 (항상 실행)
        demo_tiktoken()
        
        # 2. 전처리 데모 (항상 실행)
        demo_preprocessing()
        
        # 3. 임베딩 데모
        if mode != "demo" or os.getenv("OPENAI_API_KEY"):
            demo_embeddings()
        else:
            print("\n[SKIP] 실습 3: OPENAI_API_KEY가 없어 건너뜁니다.")
        
        # 4. 유사도 계산 데모
        if mode != "demo" or os.getenv("OPENAI_API_KEY"):
            demo_similarity()
        else:
            print("\n[SKIP] 실습 4: OPENAI_API_KEY가 없어 건너뜁니다.")
        
        # 5. 검색 엔진 데모
        if mode != "demo" or os.getenv("OPENAI_API_KEY"):
            demo_search_engine()
        else:
            print("\n[SKIP] 실습 5: OPENAI_API_KEY가 없어 건너뜁니다.")
        
        # 6~9: 심화 실습 (quick 모드에서는 건너뜀)
        if mode != "quick":
            # 6. 임베딩 시각화
            demo_embedding_visualization()
            
            # 7. Sentence Transformers
            demo_sentence_transformers()
            
            # 8. 임베딩 모델 비교
            demo_embedding_model_comparison()
            
            # 9. 한글-영어 임베딩 비교 (새로 추가!)
            if mode == "run" or os.getenv("OPENAI_API_KEY"):
                demo_korean_english_comparison()
            else:
                print("\n[SKIP] 실습 9: --run 모드 또는 OPENAI_API_KEY 필요")
        
        # 완료 메시지
        print("\n" + "="*80)
        print("[OK] 모든 실습 완료!")
        print("="*80)
        
        print("\n[INFO] 오늘 배운 내용 요약:")
        print("  ┌─────────────────────────────────────────────────────")
        print("  │ 1. 토큰: GPT가 텍스트를 처리하는 단위 (비용 계산 기준)")
        print("  │ 2. 전처리: BM25엔 필수, 임베딩엔 불필요 ⚠️")
        print("  │ 3. 임베딩: 텍스트를 숫자 벡터로 변환")
        print("  │ 4. 코사인 유사도: 벡터 간 유사성 측정 (-1 ~ 1)")
        print("  │ 5. 의미 검색: 임베딩 기반으로 유사 문서 찾기")
        
        if mode != "quick":
            print("  │ 6. 시각화: t-SNE/UMAP으로 벡터 공간 이해")
            print("  │ 7. 오픈소스: Sentence Transformers로 비용 절감")
            print("  │ 8. 모델 선택: 품질/비용 Trade-off 이해")
            print("  │ 9. 다국어: 한글-영어 임베딩 정렬 품질 확인 🆕")
        
        print("  └─────────────────────────────────────────────────────")
        
        # 생성된 파일 안내
        if mode == "run":
            output_dir = Path(__file__).parent
            print(f"\n[FILE] 생성된 파일:")
            
            tsne_file = output_dir / "embedding_tsne_demo.png"
            if tsne_file.exists():
                print(f"   - {tsne_file.name} : t-SNE 시각화")
        
        print("\n[TIP] 다음 단계:")
        print("   - lab02/vector_db.py : Vector DB (ChromaDB)로 대용량 검색")
        print("   - lab03/rag_basic.py : RAG 시스템 구축 (검색 + LLM)")
        
        if mode == "quick":
            print("\n[TIP] 심화 실습을 원하시면:")
            print("   python nlp_basics.py --run")
        
    except Exception as e:
        print(f"\n[X] 오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
