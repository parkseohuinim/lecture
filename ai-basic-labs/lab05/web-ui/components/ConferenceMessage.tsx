'use client';

import { ConferenceMessage as MessageType } from '@/hooks/useConference';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

interface ConferenceMessageProps {
  message: MessageType;
}

export function ConferenceMessage({ message }: ConferenceMessageProps) {
  // 메시지 타입별 처리
  if (message.type === 'conference_start') {
    return (
      <div className="flex items-center space-x-3 py-3 px-4 bg-blue-50 dark:bg-blue-900/20 rounded-xl border border-blue-200 dark:border-blue-800">
        <svg className="w-5 h-5 text-blue-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
        <span className="text-sm font-medium text-blue-700 dark:text-blue-300">
          회의가 시작되었습니다: {message.pattern}
        </span>
      </div>
    );
  }

  if (message.type === 'conference_complete') {
    return (
      <div className="flex items-center space-x-3 py-3 px-4 bg-green-50 dark:bg-green-900/20 rounded-xl border border-green-200 dark:border-green-800">
        <svg className="w-5 h-5 text-green-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
        <span className="text-sm font-medium text-green-700 dark:text-green-300">
          회의가 완료되었습니다
        </span>
      </div>
    );
  }

  // 병렬 처리 시작 메시지
  if (message.type === 'parallel_start') {
    return (
      <div className="relative">
        {/* 상단 구분선 */}
        <div className="flex items-center gap-3 mb-4">
          <div className="flex-1 h-[2px] bg-gradient-to-r from-transparent via-purple-400 to-purple-500"></div>
          <span className="text-xs font-bold text-purple-500 dark:text-purple-400 tracking-wider uppercase">
            ━━ 병렬 처리 시작 ━━
          </span>
          <div className="flex-1 h-[2px] bg-gradient-to-l from-transparent via-purple-400 to-purple-500"></div>
        </div>

        {/* 병렬 블록 헤더 */}
        <div className="bg-gradient-to-r from-purple-50 via-indigo-50 to-purple-50 dark:from-purple-900/30 dark:via-indigo-900/30 dark:to-purple-900/30 rounded-2xl p-5 border-2 border-purple-300 dark:border-purple-700 shadow-lg">
          <div className="flex items-center space-x-3 mb-3">
            <div className="flex items-center justify-center w-10 h-10 bg-purple-500 rounded-full shadow-lg">
              <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 10h16M4 14h16M4 18h16" />
              </svg>
            </div>
            <div>
              <h3 className="text-lg font-bold text-purple-800 dark:text-purple-200">
                {message.group_title || '병렬 분석'}
              </h3>
              <p className="text-xs text-purple-600 dark:text-purple-400">
                {message.group_description || '동시 처리 중'}
              </p>
            </div>
          </div>
          
          {/* 안건 표시 */}
          {message.topic && (
            <div className="bg-white/60 dark:bg-gray-800/60 rounded-xl p-3 mb-3">
              <p className="text-xs text-gray-500 dark:text-gray-400 mb-1">공통 안건</p>
              <p className="text-sm font-semibold text-gray-800 dark:text-gray-200">
                &ldquo;{message.topic}&rdquo;
              </p>
            </div>
          )}
          
          {/* 병렬 노드 목록 */}
          <div className="flex flex-wrap gap-2">
            {message.parallel_nodes?.map((node, idx) => (
              <div 
                key={idx}
                className="flex items-center space-x-1.5 px-3 py-1.5 bg-white/70 dark:bg-gray-700/70 rounded-full border border-purple-200 dark:border-purple-600"
              >
                <span className="text-sm">{getNodeIcon(node)}</span>
                <span className="text-xs font-medium text-gray-700 dark:text-gray-300">
                  {getNodeDisplayName(node)}
                </span>
                <span className="text-xs text-purple-500 animate-pulse">▶</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    );
  }

  // 병렬 처리 종료 메시지
  if (message.type === 'parallel_end') {
    return (
      <div className="relative">
        {/* 병렬 종료 블록 */}
        <div className="bg-gradient-to-r from-green-50 via-emerald-50 to-green-50 dark:from-green-900/30 dark:via-emerald-900/30 dark:to-green-900/30 rounded-2xl p-4 border-2 border-green-300 dark:border-green-700 shadow-lg">
          <div className="flex items-center space-x-3">
            <div className="flex items-center justify-center w-10 h-10 bg-green-500 rounded-full shadow-lg">
              <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
              </svg>
            </div>
            <div className="flex-1">
              <h3 className="text-sm font-bold text-green-800 dark:text-green-200">
                병렬 분석 완료 → 결과 수집
              </h3>
              <p className="text-xs text-green-600 dark:text-green-400">
                {message.completed_nodes?.length || 0}개 에이전트 분석 완료
              </p>
            </div>
          </div>
        </div>
        
        {/* 화살표 & 다음 단계 표시 */}
        <div className="flex flex-col items-center my-3">
          <div className="w-[2px] h-6 bg-gradient-to-b from-green-400 to-blue-500"></div>
          <svg className="w-6 h-6 text-blue-500 -mt-1" fill="currentColor" viewBox="0 0 24 24">
            <path d="M12 16l-6-6h12l-6 6z" />
          </svg>
        </div>
        
        {/* 다음 노드 안내 */}
        {message.next_node && (
          <div className="flex items-center justify-center space-x-2 px-4 py-2 bg-blue-100 dark:bg-blue-900/40 rounded-xl border border-blue-300 dark:border-blue-700">
            <span className="text-lg">{getNodeIcon(message.next_node)}</span>
            <span className="text-sm font-semibold text-blue-700 dark:text-blue-300">
              {getNodeDisplayName(message.next_node)} - 최종 결정
            </span>
          </div>
        )}
        
        {/* 하단 구분선 */}
        <div className="flex items-center gap-3 mt-4">
          <div className="flex-1 h-[2px] bg-gradient-to-r from-transparent via-green-400 to-green-500"></div>
          <span className="text-xs font-bold text-green-500 dark:text-green-400 tracking-wider uppercase">
            ━━━━━━━━━━━━━
          </span>
          <div className="flex-1 h-[2px] bg-gradient-to-l from-transparent via-green-400 to-green-500"></div>
        </div>
      </div>
    );
  }

  if (message.type === 'error' || message.type === 'conference_error') {
    return (
      <div className="flex items-start space-x-3 py-3 px-4 bg-red-50 dark:bg-red-900/20 rounded-xl border border-red-200 dark:border-red-800">
        <svg className="w-5 h-5 text-red-500 flex-shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
        <div>
          <p className="text-sm font-medium text-red-700 dark:text-red-300 mb-1">
            오류 발생
          </p>
          <p className="text-xs text-red-600 dark:text-red-400">
            {message.error || '알 수 없는 오류'}
          </p>
        </div>
      </div>
    );
  }

  // HITL 세션 시작 메시지
  if (message.type === 'hitl_session_start') {
    return (
      <div className="flex items-center space-x-3 py-4 px-5 bg-gradient-to-r from-amber-50 to-yellow-50 dark:from-amber-900/30 dark:to-yellow-900/30 rounded-2xl border-2 border-amber-300 dark:border-amber-700 shadow-lg">
        <div className="flex items-center justify-center w-12 h-12 bg-amber-500 rounded-full shadow-lg">
          <span className="text-2xl">👤</span>
        </div>
        <div>
          <h3 className="text-lg font-bold text-amber-800 dark:text-amber-200">
            HITL 세션 시작
          </h3>
          <p className="text-sm text-amber-600 dark:text-amber-400">
            Human-in-the-Loop 워크플로우가 시작되었습니다
          </p>
          <p className="text-xs text-amber-500 dark:text-amber-500 mt-1">
            세션 ID: {message.session_id} | 최대 수정: {message.max_revisions}회
          </p>
        </div>
      </div>
    );
  }

  // HITL 사용자 입력 대기 메시지 (이건 ConferenceRoom에서 별도 UI로 표시)
  if (message.type === 'hitl_awaiting_input') {
    return (
      <div className="flex items-center space-x-3 py-4 px-5 bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/30 dark:to-indigo-900/30 rounded-2xl border-2 border-blue-300 dark:border-blue-700 shadow-lg animate-pulse">
        <div className="flex items-center justify-center w-12 h-12 bg-blue-500 rounded-full shadow-lg">
          <span className="text-2xl">⏳</span>
        </div>
        <div>
          <h3 className="text-lg font-bold text-blue-800 dark:text-blue-200">
            사용자 결정 대기 중
          </h3>
          <p className="text-sm text-blue-600 dark:text-blue-400">
            제안서를 검토하고 결정해주세요 (승인 / 수정 요청 / 거부)
          </p>
          <p className="text-xs text-blue-500 dark:text-blue-500 mt-1">
            수정 횟수: {message.revision_count || 0} / {message.max_revisions || 3}회
          </p>
        </div>
      </div>
    );
  }

  // HITL 사용자 결정 메시지
  if (message.type === 'hitl_user_decision') {
    return (
      <div className="flex items-start space-x-3">
        <div className="flex-shrink-0 w-10 h-10 rounded-full bg-gradient-to-r from-green-500 to-emerald-500 flex items-center justify-center shadow-lg">
          <span className="text-lg">👤</span>
        </div>
        <div className="flex-1 min-w-0">
          <div className="rounded-2xl p-4 shadow-sm border bg-gradient-to-br from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 border-green-200 dark:border-green-700">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm font-semibold text-gray-900 dark:text-white">
                👤 사용자 결정
              </span>
              <span className="text-xs text-gray-500 dark:text-gray-400">
                {new Date().toLocaleTimeString('ko-KR', {
                  hour: '2-digit',
                  minute: '2-digit',
                  second: '2-digit'
                })}
              </span>
            </div>
            <div className="prose prose-sm dark:prose-invert max-w-none">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                {message.content || ''}
              </ReactMarkdown>
            </div>
          </div>
        </div>
      </div>
    );
  }

  // 에이전트 메시지
  if (message.type === 'agent_message' && message.node && message.content) {
    const nodeDisplayName = getNodeDisplayName(message.node);
    const nodeIcon = getNodeIcon(message.node);
    const nodeColor = getNodeColor(message.node);
    const isParallel = message.is_parallel;

    return (
      <div className={`group relative ${isParallel ? 'pl-4 border-l-4 border-purple-300 dark:border-purple-600' : ''}`}>
        {/* 병렬 처리 배지 */}
        {isParallel && (
          <div className="absolute -left-[10px] top-4 flex items-center justify-center w-5 h-5 bg-purple-500 rounded-full text-white text-[10px] font-bold shadow-md">
            ⚡
          </div>
        )}
        
        {/* 에이전트 아바타 */}
        <div className="flex items-start space-x-3">
          <div className={`flex-shrink-0 w-10 h-10 rounded-full ${nodeColor} flex items-center justify-center shadow-lg ${isParallel ? 'ring-2 ring-purple-400 ring-offset-2 dark:ring-offset-gray-800' : ''}`}>
            <span className="text-lg">{nodeIcon}</span>
          </div>

          {/* 메시지 내용 */}
          <div className="flex-1 min-w-0">
            <div className={`rounded-2xl p-4 shadow-sm border ${
              isParallel 
                ? 'bg-gradient-to-br from-white to-purple-50 dark:from-gray-700 dark:to-purple-900/20 border-purple-200 dark:border-purple-700' 
                : 'bg-white dark:bg-gray-700 border-gray-200 dark:border-gray-600'
            }`}>
              {/* 헤더 */}
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center space-x-2">
                  <span className="text-sm font-semibold text-gray-900 dark:text-white">
                    {nodeDisplayName}
                  </span>
                  {isParallel && (
                    <span className="px-2 py-0.5 text-[10px] font-bold bg-purple-100 dark:bg-purple-800/50 text-purple-600 dark:text-purple-300 rounded-full">
                      병렬 분석
                    </span>
                  )}
                </div>
                <span className="text-xs text-gray-500 dark:text-gray-400">
                  {new Date().toLocaleTimeString('ko-KR', {
                    hour: '2-digit',
                    minute: '2-digit',
                    second: '2-digit'
                  })}
                </span>
              </div>

              {/* 내용 (Markdown 렌더링) */}
              <div className="prose prose-sm dark:prose-invert max-w-none">
                <ReactMarkdown
                  remarkPlugins={[remarkGfm]}
                  components={{
                    p: ({ ...props }) => <p className="text-sm text-gray-700 dark:text-gray-300 mb-2" {...props} />,
                    ul: ({ ...props }) => <ul className="list-disc list-inside text-sm space-y-1" {...props} />,
                    ol: ({ ...props }) => <ol className="list-decimal list-inside text-sm space-y-1" {...props} />,
                    code: ({ inline, ...props }: { inline?: boolean } & React.HTMLAttributes<HTMLElement>) => (
                      inline ? (
                        <code className="px-1.5 py-0.5 rounded bg-gray-200 dark:bg-gray-600 font-mono text-xs" {...props} />
                      ) : (
                        <code className="block p-2 rounded-lg bg-gray-900 text-gray-100 overflow-x-auto font-mono text-xs my-2" {...props} />
                      )
                    ),
                    // 테이블 스타일링
                    table: ({ ...props }) => (
                      <div className="overflow-x-auto my-3">
                        <table className="min-w-full text-xs border-collapse border border-gray-300 dark:border-gray-600 rounded-lg overflow-hidden" {...props} />
                      </div>
                    ),
                    thead: ({ ...props }) => (
                      <thead className="bg-gray-100 dark:bg-gray-700" {...props} />
                    ),
                    tbody: ({ ...props }) => (
                      <tbody className="divide-y divide-gray-200 dark:divide-gray-600" {...props} />
                    ),
                    tr: ({ ...props }) => (
                      <tr className="hover:bg-gray-50 dark:hover:bg-gray-700/50 transition-colors" {...props} />
                    ),
                    th: ({ ...props }) => (
                      <th className="px-3 py-2 text-left text-xs font-semibold text-gray-700 dark:text-gray-200 border border-gray-300 dark:border-gray-600 bg-gray-100 dark:bg-gray-700" {...props} />
                    ),
                    td: ({ ...props }) => (
                      <td className="px-3 py-2 text-xs text-gray-600 dark:text-gray-300 border border-gray-300 dark:border-gray-600" {...props} />
                    ),
                  }}
                >
                  {message.content}
                </ReactMarkdown>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  // 기타 메시지
  return (
    <div className="py-2 px-4 bg-gray-50 dark:bg-gray-800/50 rounded-lg">
      <pre className="text-xs text-gray-600 dark:text-gray-400 overflow-x-auto">
        {JSON.stringify(message, null, 2)}
      </pre>
    </div>
  );
}

// 노드 이름 매핑
function getNodeDisplayName(node: string): string {
  const names: Record<string, string> = {
    'summarizer': '요약 전문가',
    'analyzer': '분석 전문가',
    'validator': '검증 전문가',
    'planner': 'Planner',
    'executor': 'Executor',
    'summarizer_node': 'Summarizer',
    'pm': 'PM',
    'developer': 'Developer',
    'designer': 'Designer',
    'qa': 'QA Engineer',
    'leader': 'Team Leader',
    'manager_delegate': 'Manager (분배)',
    'worker1': 'Worker 1',
    'worker2': 'Worker 2',
    'worker3': 'Worker 3',
    'manager_integrate': 'Manager (통합)',
    'proposer_initial': 'Proposer (제안)',
    'critic': 'Critic (비평)',
    'proposer_refine': 'Proposer (개선)',
    'judge': 'Judge (판결)',
    'agent1': '비용 최적화 전문가',
    'agent2': '성능 최우선 전문가',
    'agent3': '보안 최우선 전문가',
    'agent4': '속도 최우선 전문가',
    'agent5': '자동화 최우선 전문가',
    'selector': 'Market Selector',
    // Reflection 패턴
    'generator': 'Generator (생성)',
    'reflector': 'Reflector (평가)',
    'finalizer': 'Finalizer (완료)',
    // Routing 패턴
    'router': 'Router (라우터)',
    'technical': '기술 전문가',
    'business': '비즈니스 전문가',
    'creative': '크리에이티브 전문가',
    'general': '일반 어시스턴트',
    'aggregator': 'Aggregator (통합)',
    // HITL 패턴
    'proposal_generator': 'Agent (제안)',
    'human_gate': 'Human Gate (검토)',
    'human_review': 'Human Review (검토)',
    // Routing 추가 (보안 전문가)
    'security': '보안 전문가',
  };

  return names[node] || node;
}

// 노드 아이콘
function getNodeIcon(node: string): string {
  const icons: Record<string, string> = {
    'summarizer': '📝',
    'analyzer': '🔍',
    'validator': '✅',
    'planner': '📋',
    'executor': '⚙️',
    'pm': '👔',
    'developer': '💻',
    'designer': '🎨',
    'qa': '🔬',
    'leader': '👨‍💼',
    'manager_delegate': '👨‍💼',
    'manager_integrate': '👨‍💼',
    'worker1': '👷',
    'worker2': '👷',
    'worker3': '👷',
    'proposer_initial': '💡',
    'proposer_refine': '💡',
    'critic': '🔍',
    'judge': '⚖️',
    'agent1': '💰',
    'agent2': '⚡',
    'agent3': '🔒',
    'agent4': '🚀',
    'agent5': '🤖',
    'selector': '🏆',
    // Reflection 패턴
    'generator': '📝',
    'reflector': '🔍',
    'finalizer': '🎯',
    // Routing 패턴
    'router': '🔀',
    'technical': '💻',
    'business': '💼',
    'creative': '🎨',
    'general': '💬',
    'aggregator': '📊',
    // HITL 패턴
    'proposal_generator': '📋',
    'human_gate': '👤',
    'human_review': '👤',
    // Routing 추가 (보안 전문가)
    'security': '🔒',
  };

  return icons[node] || '🤖';
}

// 노드 색상
function getNodeColor(node: string): string {
  if (node.includes('summarizer')) return 'bg-gradient-to-r from-blue-500 to-blue-600';
  if (node.includes('analyzer')) return 'bg-gradient-to-r from-purple-500 to-purple-600';
  if (node.includes('validator')) return 'bg-gradient-to-r from-green-500 to-green-600';
  if (node.includes('planner')) return 'bg-gradient-to-r from-indigo-500 to-indigo-600';
  if (node.includes('executor')) return 'bg-gradient-to-r from-orange-500 to-orange-600';
  if (node === 'pm') return 'bg-gradient-to-r from-blue-500 to-blue-600';
  if (node === 'developer') return 'bg-gradient-to-r from-green-500 to-green-600';
  if (node === 'designer') return 'bg-gradient-to-r from-pink-500 to-pink-600';
  if (node === 'qa') return 'bg-gradient-to-r from-purple-500 to-purple-600';
  if (node.includes('leader') || node.includes('manager')) return 'bg-gradient-to-r from-red-500 to-red-600';
  if (node.includes('worker')) return 'bg-gradient-to-r from-teal-500 to-teal-600';
  if (node.includes('proposer')) return 'bg-gradient-to-r from-yellow-500 to-yellow-600';
  if (node === 'critic') return 'bg-gradient-to-r from-red-500 to-red-600';
  if (node === 'judge') return 'bg-gradient-to-r from-gray-600 to-gray-700';
  if (node === 'agent1') return 'bg-gradient-to-r from-emerald-500 to-emerald-600';  // 비용 - 초록
  if (node === 'agent2') return 'bg-gradient-to-r from-blue-500 to-blue-600';      // 성능 - 파랑
  if (node === 'agent3') return 'bg-gradient-to-r from-red-500 to-red-600';        // 보안 - 빨강
  if (node === 'agent4') return 'bg-gradient-to-r from-orange-500 to-orange-600';  // 속도 - 주황
  if (node === 'agent5') return 'bg-gradient-to-r from-purple-500 to-purple-600';  // 자동화 - 보라
  if (node === 'selector') return 'bg-gradient-to-r from-yellow-500 to-yellow-600';
  // Reflection 패턴
  if (node === 'generator') return 'bg-gradient-to-r from-cyan-500 to-cyan-600';
  if (node === 'reflector') return 'bg-gradient-to-r from-amber-500 to-amber-600';
  if (node === 'finalizer') return 'bg-gradient-to-r from-emerald-500 to-emerald-600';
  // Routing 패턴
  if (node === 'router') return 'bg-gradient-to-r from-violet-500 to-violet-600';
  if (node === 'technical') return 'bg-gradient-to-r from-blue-500 to-blue-600';
  if (node === 'business') return 'bg-gradient-to-r from-emerald-500 to-emerald-600';
  if (node === 'creative') return 'bg-gradient-to-r from-pink-500 to-pink-600';
  if (node === 'general') return 'bg-gradient-to-r from-gray-500 to-gray-600';
  if (node === 'aggregator') return 'bg-gradient-to-r from-indigo-500 to-indigo-600';
  // HITL 패턴
  if (node === 'proposal_generator') return 'bg-gradient-to-r from-blue-500 to-blue-600';
  if (node === 'human_gate') return 'bg-gradient-to-r from-amber-500 to-amber-600';
  if (node === 'human_review') return 'bg-gradient-to-r from-amber-500 to-amber-600';
  // Routing 추가 (보안 전문가)
  if (node === 'security') return 'bg-gradient-to-r from-red-500 to-red-600';

  return 'bg-gradient-to-r from-gray-500 to-gray-600';
}

