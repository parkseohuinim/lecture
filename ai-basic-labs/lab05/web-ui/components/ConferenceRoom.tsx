'use client';

import { useState, useEffect, useRef, useMemo } from 'react';
import { useConference, ConferenceMessage as MessageType, HITLDecision } from '@/hooks/useConference';
import { ConferenceMessage } from './ConferenceMessage';

interface Pattern {
  id: string;
  name: string;
  description: string;
  icon: string;
  difficulty: string;
}

export function ConferenceRoom() {
  const [patterns, setPatterns] = useState<Pattern[]>([]);
  const [selectedPattern, setSelectedPattern] = useState<string>('sequential');
  const [topic, setTopic] = useState<string>('');
  const [maxRounds, setMaxRounds] = useState<number>(3);
  const [numAgents, setNumAgents] = useState<number>(5);
  const [maxRevisions, setMaxRevisions] = useState<number>(3);
  
  // HITL 결정 UI 상태
  const [hitlFeedback, setHitlFeedback] = useState<string>('');
  
  const { 
    getMessages, 
    clearMessages, 
    isConnected, 
    isRunning, 
    startConference,
    stopConference,  // 회의 중지 기능
    // HITL 전용
    hitlAwaitingInput,
    hitlRevisionCount,
    hitlMaxRevisions,
    submitHITLDecision
  } = useConference();
  
  // 스크롤 자동 이동을 위한 ref
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const messagesContainerRef = useRef<HTMLDivElement>(null);
  
  // 스마트 스크롤 상태
  const [isUserScrolledUp, setIsUserScrolledUp] = useState(false);
  const prevMessageCountRef = useRef(0);
  
  // 현재 선택된 패턴의 메시지
  const messages = getMessages(selectedPattern);

  // 패턴 목록 로드
  useEffect(() => {
    let isMounted = true;
    const controller = new AbortController();
    
    const loadPatterns = async () => {
      try {
        const response = await fetch('http://localhost:8000/api/conference/patterns', {
          signal: controller.signal
        });
        const data = await response.json();
        
        // 컴포넌트가 마운트된 상태일 때만 상태 업데이트
        if (isMounted && data.success) {
          setPatterns(data.patterns);
        }
      } catch (err) {
        // AbortError는 정상적인 취소이므로 무시
        if (isMounted && (err as Error).name !== 'AbortError') {
          console.error('패턴 로드 실패:', err);
        }
      }
    };
    
    loadPatterns();
    
    // Cleanup: 컴포넌트 언마운트 시
    return () => {
      isMounted = false;
      controller.abort();
    };
  }, []);

  const handleStart = () => {
    if (!topic.trim()) {
      alert('주제를 입력해주세요');
      return;
    }

    // HITL 피드백 초기화
    setHitlFeedback('');

    startConference({
      pattern: selectedPattern,
      topic: topic,
      max_rounds: maxRounds,
      num_agents: numAgents,
      max_revisions: maxRevisions
    });
  };

  // HITL 결정 핸들러
  const handleHITLDecision = (decision: 'approve' | 'revision' | 'reject') => {
    submitHITLDecision({
      decision,
      feedback: hitlFeedback
    });
    setHitlFeedback('');
  };

  // 엔터키로 회의 시작
  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter' && !isRunning && topic.trim()) {
      e.preventDefault();
      handleStart();
    }
  };

  // 사용자 스크롤 감지 - 맨 아래에서 벗어나면 자동 스크롤 비활성화
  const handleScroll = () => {
    const container = messagesContainerRef.current;
    if (!container) return;
    
    // 맨 아래에서 100px 이내면 "맨 아래"로 간주
    const isAtBottom = container.scrollHeight - container.scrollTop - container.clientHeight < 100;
    setIsUserScrolledUp(!isAtBottom);
  };

  // 스마트 스크롤: 사용자가 위로 스크롤하지 않았으면 항상 스크롤
  // (병렬 노드는 스트리밍 비활성화되어 있으므로 토큰 스트리밍 시에도 스크롤해도 OK)
  useEffect(() => {
    // 조건: 사용자가 위로 스크롤하지 않았고, 메시지가 있는 경우
    if (!isUserScrolledUp && messages.length > 0) {
      messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }
    
    prevMessageCountRef.current = messages.length;
  }, [messages, isUserScrolledUp]);

  const getDifficultyColor = (difficulty: string) => {
    switch (difficulty) {
      case 'easy':
        return 'text-green-600 dark:text-green-400';
      case 'medium':
        return 'text-yellow-600 dark:text-yellow-400';
      case 'hard':
        return 'text-red-600 dark:text-red-400';
      default:
        return 'text-gray-600 dark:text-gray-400';
    }
  };

  // 병렬 메시지 그룹화 렌더링
  const renderGroupedMessages = (messages: MessageType[]) => {
    const result: JSX.Element[] = [];
    let isInParallelBlock = false;
    let parallelMessages: MessageType[] = [];
    let parallelStartMsg: MessageType | null = null;
    let messageKey = 0;

    for (let i = 0; i < messages.length; i++) {
      const msg = messages[i];

      // parallel_start 발견
      if (msg.type === 'parallel_start') {
        isInParallelBlock = true;
        parallelStartMsg = msg;
        parallelMessages = [];
        // parallel_start 메시지 렌더링
        result.push(<ConferenceMessage key={`msg-${messageKey++}`} message={msg} />);
        continue;
      }

      // parallel_end 발견
      if (msg.type === 'parallel_end') {
        isInParallelBlock = false;
        
        // 병렬 메시지들을 그룹으로 렌더링
        if (parallelMessages.length > 0) {
          result.push(
            <div key={`parallel-group-${messageKey++}`} className="relative">
              {/* 병렬 그룹 컨테이너 */}
              <div className="ml-2 pl-4 border-l-4 border-purple-400 dark:border-purple-600 space-y-3 py-2 bg-gradient-to-r from-purple-50/50 to-transparent dark:from-purple-900/10 dark:to-transparent rounded-r-xl">
                {parallelMessages.map((parallelMsg, idx) => (
                  <ConferenceMessage key={`parallel-${messageKey}-${idx}`} message={parallelMsg} />
                ))}
              </div>
            </div>
          );
        }
        
        // parallel_end 메시지 렌더링
        result.push(<ConferenceMessage key={`msg-${messageKey++}`} message={msg} />);
        parallelMessages = [];
        parallelStartMsg = null;
        continue;
      }

      // 병렬 블록 내 메시지 (agent_message와 agent_streaming 모두 포함)
      if (isInParallelBlock && (msg.type === 'agent_message' || msg.type === 'agent_streaming')) {
        parallelMessages.push(msg);
        continue;
      }

      // 일반 메시지
      result.push(<ConferenceMessage key={`msg-${messageKey++}`} message={msg} />);
    }

    // 남은 병렬 메시지가 있으면 렌더링 (스트리밍 중인 경우)
    if (isInParallelBlock && parallelMessages.length > 0) {
      result.push(
        <div key={`parallel-group-${messageKey++}`} className="relative">
          {/* 진행 중인 병렬 그룹 컨테이너 */}
          <div className="ml-2 pl-4 border-l-4 border-purple-400 dark:border-purple-600 border-dashed space-y-3 py-2 bg-gradient-to-r from-purple-50/50 to-transparent dark:from-purple-900/10 dark:to-transparent rounded-r-xl animate-pulse">
            {parallelMessages.map((parallelMsg, idx) => (
              <ConferenceMessage key={`parallel-${messageKey}-${idx}`} message={parallelMsg} />
            ))}
            {/* 진행 중 표시 */}
            <div className="flex items-center space-x-2 text-sm text-purple-500 dark:text-purple-400 pl-4">
              <svg className="animate-spin h-4 w-4" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
              <span>병렬 분석 진행 중...</span>
            </div>
          </div>
        </div>
      );
    }

    return result;
  };

  return (
    <div className="glass rounded-3xl p-6 shadow-2xl">
      {/* 패턴 선택 */}
      <div className="mb-6">
        <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-4">
          멀티 에이전트 패턴 선택
        </h3>
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
          {patterns.map((pattern) => (
            <button
              key={pattern.id}
              onClick={() => setSelectedPattern(pattern.id)}
              disabled={isRunning}
              className={`p-4 rounded-xl border-2 transition-all duration-200 text-left ${
                selectedPattern === pattern.id
                  ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
                  : 'border-gray-200 dark:border-gray-700 hover:border-blue-300 dark:hover:border-blue-700'
              } ${isRunning ? 'opacity-50 cursor-not-allowed' : ''}`}
            >
              <div className="flex items-start space-x-3">
                <span className="text-3xl">{pattern.icon}</span>
                <div className="flex-1 min-w-0">
                  <h4 className="font-semibold text-gray-900 dark:text-white text-sm mb-1">
                    {pattern.name}
                  </h4>
                  <p className="text-xs text-gray-600 dark:text-gray-400 line-clamp-2">
                    {pattern.description}
                  </p>
                  <span className={`text-xs font-medium mt-1 inline-block ${getDifficultyColor(pattern.difficulty)}`}>
                    {pattern.difficulty === 'easy' && '초급'}
                    {pattern.difficulty === 'medium' && '중급'}
                    {pattern.difficulty === 'hard' && '고급'}
                  </span>
                </div>
              </div>
            </button>
          ))}
        </div>
      </div>

      {/* 설정 */}
      <div className="mb-6 space-y-4">
        {/* 주제 입력 */}
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            회의 주제 <span className="text-gray-500 text-xs">(Enter로 시작)</span>
          </label>
          <input
            type="text"
            value={topic}
            onChange={(e) => setTopic(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={isRunning}
            placeholder="예: AI 멀티 에이전트 시스템의 장단점"
            className="w-full px-4 py-3 rounded-xl border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 text-gray-900 dark:text-white placeholder-gray-400 dark:placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:opacity-50"
          />
        </div>

        {/* 패턴별 추가 설정 */}
        {selectedPattern === 'debate' && (
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              토론 라운드 수
            </label>
            <input
              type="number"
              value={maxRounds}
              onChange={(e) => setMaxRounds(parseInt(e.target.value) || 3)}
              disabled={isRunning}
              min={1}
              max={10}
              className="w-full px-4 py-3 rounded-xl border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 text-gray-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:opacity-50"
            />
          </div>
        )}

        {selectedPattern === 'swarm' && (
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              에이전트 수
            </label>
            <input
              type="number"
              value={numAgents}
              onChange={(e) => setNumAgents(parseInt(e.target.value) || 5)}
              disabled={isRunning}
              min={2}
              max={10}
              className="w-full px-4 py-3 rounded-xl border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 text-gray-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:opacity-50"
            />
          </div>
        )}

        {selectedPattern === 'hitl' && (
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              최대 수정 횟수
            </label>
            <input
              type="number"
              value={maxRevisions}
              onChange={(e) => setMaxRevisions(parseInt(e.target.value) || 3)}
              disabled={isRunning}
              min={1}
              max={10}
              className="w-full px-4 py-3 rounded-xl border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 text-gray-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:opacity-50"
            />
            <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
              수정 요청 가능한 최대 횟수입니다. 이 횟수를 초과하면 자동 승인됩니다.
            </p>
          </div>
        )}
      </div>

      {/* 액션 버튼들 */}
      <div className="mb-6 flex space-x-3">
        {/* 시작/중지 버튼 */}
        {isRunning ? (
          <button
            onClick={stopConference}
            className="flex-1 py-4 rounded-xl bg-gradient-to-r from-red-500 to-orange-500 hover:from-red-600 hover:to-orange-600 text-white font-semibold shadow-lg transform hover:scale-105 transition-all duration-200"
          >
            <div className="flex items-center justify-center space-x-2">
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 10a1 1 0 011-1h4a1 1 0 011 1v4a1 1 0 01-1 1h-4a1 1 0 01-1-1v-4z" />
              </svg>
              <span>회의 중지</span>
            </div>
          </button>
        ) : (
          <button
            onClick={handleStart}
            disabled={!topic.trim()}
            className="flex-1 py-4 rounded-xl bg-gradient-to-r from-blue-500 to-purple-500 hover:from-blue-600 hover:to-purple-600 text-white font-semibold shadow-lg transform hover:scale-105 transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none"
          >
            <div className="flex items-center justify-center space-x-2">
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              <span>회의 시작</span>
            </div>
          </button>
        )}

        {/* 대화 내용 비우기 버튼 */}
        <button
          onClick={() => {
            if (confirm(`${selectedPattern} 패턴의 대화 내용을 모두 삭제하시겠습니까?`)) {
              clearMessages(selectedPattern);
            }
          }}
          disabled={isRunning || messages.length === 0}
          className="px-6 py-4 rounded-xl bg-gray-200 dark:bg-gray-700 hover:bg-gray-300 dark:hover:bg-gray-600 text-gray-700 dark:text-gray-300 font-semibold shadow-lg transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
          title="현재 패턴의 대화 내용 비우기"
        >
          <div className="flex items-center justify-center space-x-2">
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
            </svg>
            <span>비우기</span>
          </div>
        </button>
      </div>

      {/* 메시지 영역 */}
      <div className="relative">
      <div 
        ref={messagesContainerRef}
        onScroll={handleScroll}
        className="bg-white/50 dark:bg-gray-800/50 rounded-2xl p-4 min-h-[400px] max-h-[600px] overflow-y-auto backdrop-blur-sm border border-gray-200/50 dark:border-gray-700/50 scroll-smooth">
        {messages.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-center py-12">
            <svg className="w-16 h-16 text-gray-400 dark:text-gray-600 mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z" />
            </svg>
            <h3 className="text-lg font-semibold text-gray-700 dark:text-gray-300 mb-2">
              회의를 시작해보세요
            </h3>
            <p className="text-sm text-gray-500 dark:text-gray-400 max-w-md">
              패턴을 선택하고 주제를 입력한 후 "회의 시작" 버튼을 누르면,
              여러 AI 에이전트들이 협업하여 토론하는 과정을 실시간으로 볼 수 있습니다.
            </p>
          </div>
        ) : (
          <div className="space-y-4">
            {renderGroupedMessages(messages)}
            
            {/* HITL 결정 UI */}
            {selectedPattern === 'hitl' && hitlAwaitingInput && (
              <div className="mt-6 p-6 bg-gradient-to-br from-amber-50 via-yellow-50 to-orange-50 dark:from-amber-900/30 dark:via-yellow-900/30 dark:to-orange-900/30 rounded-2xl border-2 border-amber-400 dark:border-amber-600 shadow-xl">
                {/* 헤더 */}
                <div className="flex items-center space-x-3 mb-4">
                  <div className="flex items-center justify-center w-12 h-12 bg-amber-500 rounded-full shadow-lg animate-pulse">
                    <span className="text-2xl">👤</span>
                  </div>
                  <div>
                    <h3 className="text-lg font-bold text-amber-800 dark:text-amber-200">
                      결정이 필요합니다
                    </h3>
                    <p className="text-sm text-amber-600 dark:text-amber-400">
                      제안서를 검토하고 결정해주세요 (수정 {hitlRevisionCount}/{hitlMaxRevisions}회)
                    </p>
                  </div>
                </div>

                {/* 피드백 입력 */}
                <div className="mb-4">
                  <label className="block text-sm font-medium text-amber-700 dark:text-amber-300 mb-2">
                    피드백 (선택사항 - 수정 요청 시 필수)
                  </label>
                  <textarea
                    value={hitlFeedback}
                    onChange={(e) => setHitlFeedback(e.target.value)}
                    placeholder="수정이 필요한 경우 구체적인 피드백을 입력해주세요..."
                    rows={3}
                    className="w-full px-4 py-3 rounded-xl border border-amber-300 dark:border-amber-600 bg-white dark:bg-gray-800 text-gray-900 dark:text-white placeholder-gray-400 dark:placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-amber-500 focus:border-transparent resize-none"
                  />
                </div>

                {/* 결정 버튼들 */}
                <div className="flex flex-wrap gap-3">
                  {/* 승인 버튼 */}
                  <button
                    onClick={() => handleHITLDecision('approve')}
                    className="flex-1 min-w-[140px] py-3 px-4 rounded-xl bg-gradient-to-r from-green-500 to-emerald-500 hover:from-green-600 hover:to-emerald-600 text-white font-semibold shadow-lg transform hover:scale-105 transition-all duration-200"
                  >
                    <div className="flex items-center justify-center space-x-2">
                      <span className="text-lg">✅</span>
                      <span>승인</span>
                    </div>
                  </button>

                  {/* 수정 요청 버튼 */}
                  <button
                    onClick={() => handleHITLDecision('revision')}
                    disabled={!hitlFeedback.trim() || hitlRevisionCount >= hitlMaxRevisions}
                    className="flex-1 min-w-[140px] py-3 px-4 rounded-xl bg-gradient-to-r from-amber-500 to-yellow-500 hover:from-amber-600 hover:to-yellow-600 text-white font-semibold shadow-lg transform hover:scale-105 transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none"
                  >
                    <div className="flex items-center justify-center space-x-2">
                      <span className="text-lg">🟡</span>
                      <span>수정 요청</span>
                    </div>
                  </button>

                  {/* 거부 버튼 */}
                  <button
                    onClick={() => handleHITLDecision('reject')}
                    className="flex-1 min-w-[140px] py-3 px-4 rounded-xl bg-gradient-to-r from-red-500 to-rose-500 hover:from-red-600 hover:to-rose-600 text-white font-semibold shadow-lg transform hover:scale-105 transition-all duration-200"
                  >
                    <div className="flex items-center justify-center space-x-2">
                      <span className="text-lg">⛔</span>
                      <span>거부</span>
                    </div>
                  </button>
                </div>

                {/* 안내 메시지 */}
                <div className="mt-4 text-xs text-amber-600 dark:text-amber-400 space-y-1">
                  <p>• <strong>승인:</strong> 현재 제안을 수락하고 워크플로우를 완료합니다.</p>
                  <p>• <strong>수정 요청:</strong> 피드백을 반영하여 제안서를 다시 생성합니다. (피드백 필수)</p>
                  <p>• <strong>거부:</strong> 제안을 거부하고 워크플로우를 종료합니다.</p>
                  {hitlRevisionCount >= hitlMaxRevisions && (
                    <p className="text-red-500 font-semibold mt-2">
                      ⚠️ 최대 수정 횟수에 도달했습니다. 승인 또는 거부만 가능합니다.
                    </p>
                  )}
                </div>
              </div>
            )}
            
            {/* 스크롤 자동 이동을 위한 마커 */}
            <div ref={messagesEndRef} />
          </div>
        )}
        
        {/* 맨 아래로 이동 버튼 (사용자가 위로 스크롤했을 때만 표시) */}
        {isUserScrolledUp && messages.length > 0 && (
          <button
            onClick={() => {
              messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
              setIsUserScrolledUp(false);
            }}
            className="absolute bottom-4 right-4 p-3 rounded-full bg-blue-500 hover:bg-blue-600 text-white shadow-lg transform hover:scale-110 transition-all duration-200 z-10"
            title="맨 아래로 이동"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 14l-7 7m0 0l-7-7m7 7V3" />
            </svg>
          </button>
        )}
      </div>
      </div>

      {/* 연결 상태 */}
      {isConnected && (
        <div className="mt-4 flex items-center space-x-2 text-sm text-green-600 dark:text-green-400">
          <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
          <span>WebSocket 연결됨</span>
        </div>
      )}
    </div>
  );
}

