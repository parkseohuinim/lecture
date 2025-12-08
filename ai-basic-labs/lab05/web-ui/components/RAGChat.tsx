'use client';

import { useState, useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

interface RAGMessage {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: Date;
  sources?: RAGSource[];
  confidence?: string;
  searchMethod?: string;
}

interface RAGSource {
  content: string;
  score: number;
  rank: number;
  filename: string;
  chunk_id: number;
}

interface DocumentInfo {
  doc_id: string;
  filename: string;
  file_type: string;
  total_chunks: number;
  uploaded_at: string;
}

export function RAGChat() {
  const [messages, setMessages] = useState<RAGMessage[]>([]);
  const [documents, setDocuments] = useState<DocumentInfo[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [input, setInput] = useState('');
  const [searchMethod, setSearchMethod] = useState<'sparse' | 'dense' | 'hybrid'>('hybrid');
  const [alpha, setAlpha] = useState(0.5);
  const [useReranker, setUseReranker] = useState(true);
  const [showSettings, setShowSettings] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // 문서 목록 로드
  useEffect(() => {
    fetchDocuments();
  }, []);

  // 스크롤 to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const fetchDocuments = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/rag/documents');
      if (response.ok) {
        const data = await response.json();
        setDocuments(data);
      }
    } catch (error) {
      console.error('문서 목록 로드 실패:', error);
    }
  };

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (!files || files.length === 0) return;

    setIsUploading(true);
    
    for (const file of files) {
      try {
        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch('http://localhost:8000/api/rag/upload', {
          method: 'POST',
          body: formData
        });

        if (!response.ok) {
          const error = await response.json();
          throw new Error(error.detail || '업로드 실패');
        }

        const data = await response.json();
        
        // 시스템 메시지 추가
        setMessages(prev => [...prev, {
          id: Date.now().toString(),
          role: 'system',
          content: `📄 **${data.filename}** 업로드 완료\n- 파일 타입: ${data.file_type}\n- 청크 수: ${data.total_chunks}개`,
          timestamp: new Date()
        }]);

        await fetchDocuments();
      } catch (error) {
        setMessages(prev => [...prev, {
          id: Date.now().toString(),
          role: 'system',
          content: `❌ 업로드 실패: ${error instanceof Error ? error.message : '알 수 없는 오류'}`,
          timestamp: new Date()
        }]);
      }
    }

    setIsUploading(false);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleSendMessage = async () => {
    if (!input.trim() || isLoading) return;

    const userMessage: RAGMessage = {
      id: Date.now().toString(),
      role: 'user',
      content: input.trim(),
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    // 스트리밍 응답을 위한 임시 메시지 생성
    const assistantMessageId = (Date.now() + 1).toString();
    let streamedContent = '';
    let sourcesData: RAGSource[] = [];
    let confidenceData = '';
    let searchMethodData = '';

    // 빈 어시스턴트 메시지 먼저 추가
    setMessages(prev => [...prev, {
      id: assistantMessageId,
      role: 'assistant',
      content: '',
      timestamp: new Date(),
      sources: [],
      confidence: undefined,
      searchMethod: undefined
    }]);

    try {
      const response = await fetch('http://localhost:8000/api/rag/stream', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          question: userMessage.content,
          k: 5,
          search_method: searchMethod,
          alpha: alpha,
          use_reranker: useReranker
        })
      });

      if (!response.ok) {
        throw new Error('스트리밍 연결 실패');
      }

      const reader = response.body?.getReader();
      const decoder = new TextDecoder();

      if (!reader) {
        throw new Error('스트림 리더를 가져올 수 없습니다');
      }

      // SSE 스트림 읽기
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        const text = decoder.decode(value, { stream: true });
        const lines = text.split('\n');

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6));

              if (data.type === 'sources') {
                // 출처 정보 수신
                sourcesData = data.data.sources;
                confidenceData = data.data.confidence;
                searchMethodData = data.data.search_method;
                
                // 출처 정보로 메시지 업데이트
                setMessages(prev => prev.map(msg => 
                  msg.id === assistantMessageId
                    ? { ...msg, sources: sourcesData, confidence: confidenceData, searchMethod: searchMethodData }
                    : msg
                ));
              } else if (data.type === 'token') {
                // 토큰 수신 - 타자치듯 추가
                streamedContent += data.data;
                
                // 메시지 내용 업데이트
                setMessages(prev => prev.map(msg => 
                  msg.id === assistantMessageId
                    ? { ...msg, content: streamedContent }
                    : msg
                ));
              } else if (data.type === 'done') {
                // 스트리밍 완료
                console.log('스트리밍 완료');
              } else if (data.type === 'error') {
                throw new Error(data.data);
              }
            } catch (parseError) {
              // JSON 파싱 실패 시 무시 (불완전한 청크일 수 있음)
            }
          }
        }
      }
    } catch (error) {
      // 에러 시 메시지 업데이트
      setMessages(prev => prev.map(msg => 
        msg.id === assistantMessageId
          ? { ...msg, content: `오류 발생: ${error instanceof Error ? error.message : '알 수 없는 오류'}` }
          : msg
      ));
    } finally {
      setIsLoading(false);
    }
  };

  const handleDeleteDocument = async (docId: string) => {
    try {
      const response = await fetch(`http://localhost:8000/api/rag/documents/${docId}`, {
        method: 'DELETE'
      });

      if (response.ok) {
        await fetchDocuments();
        setMessages(prev => [...prev, {
          id: Date.now().toString(),
          role: 'system',
          content: `🗑️ 문서 삭제 완료`,
          timestamp: new Date()
        }]);
      }
    } catch (error) {
      console.error('문서 삭제 실패:', error);
    }
  };

  const handleClearAllDocuments = async () => {
    if (!confirm('모든 문서를 삭제하시겠습니까?\n이 작업은 되돌릴 수 없습니다.')) {
      return;
    }

    try {
      const response = await fetch('http://localhost:8000/api/rag/documents', {
        method: 'DELETE'
      });

      if (response.ok) {
        await fetchDocuments();
        setMessages(prev => [...prev, {
          id: Date.now().toString(),
          role: 'system',
          content: `🗑️ 모든 문서가 삭제되었습니다.`,
          timestamp: new Date()
        }]);
      }
    } catch (error) {
      console.error('전체 문서 삭제 실패:', error);
      setMessages(prev => [...prev, {
        id: Date.now().toString(),
        role: 'system',
        content: `❌ 전체 삭제 실패: ${error instanceof Error ? error.message : '알 수 없는 오류'}`,
        timestamp: new Date()
      }]);
    }
  };

  const getConfidenceColor = (confidence: string) => {
    switch (confidence) {
      case 'high': return 'text-green-500 bg-green-100 dark:bg-green-900/30';
      case 'medium': return 'text-yellow-500 bg-yellow-100 dark:bg-yellow-900/30';
      case 'low': return 'text-red-500 bg-red-100 dark:bg-red-900/30';
      default: return 'text-gray-500 bg-gray-100 dark:bg-gray-900/30';
    }
  };

  return (
    <div className="flex h-[calc(100vh-16rem)] glass rounded-3xl shadow-2xl overflow-hidden border border-white/20 dark:border-gray-700/50">
      {/* 사이드바 - 문서 목록 */}
      <div className="w-72 bg-white/30 dark:bg-gray-800/30 backdrop-blur-sm border-r border-gray-200/50 dark:border-gray-700/50 flex flex-col">
        {/* 업로드 버튼 */}
        <div className="p-4 border-b border-gray-200/50 dark:border-gray-700/50">
          <label className="flex items-center justify-center w-full px-4 py-3 bg-gradient-to-r from-emerald-500 to-teal-500 text-white rounded-xl cursor-pointer hover:from-emerald-600 hover:to-teal-600 transition-all shadow-lg hover:shadow-xl">
            <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
            </svg>
            {isUploading ? '업로드 중...' : '문서 업로드'}
            <input
              ref={fileInputRef}
              type="file"
              multiple
              accept=".pdf,.md,.markdown,.json,.txt,.text"
              onChange={handleFileUpload}
              className="hidden"
              disabled={isUploading}
            />
          </label>
          <p className="text-xs text-gray-500 dark:text-gray-400 text-center mt-2">
            PDF, MD, JSON, TXT 지원
          </p>
        </div>

        {/* 문서 목록 */}
        <div className="flex-1 overflow-y-auto p-4">
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-semibold text-gray-700 dark:text-gray-300 flex items-center">
              <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
              </svg>
              문서 ({documents.length})
            </h3>
            {/* 전체 삭제 버튼 */}
            {documents.length > 0 && (
              <button
                onClick={handleClearAllDocuments}
                className="px-2 py-1 text-xs text-red-500 hover:bg-red-100 dark:hover:bg-red-900/30 rounded-lg transition-all flex items-center"
                title="모든 문서 삭제"
              >
                <svg className="w-3.5 h-3.5 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                </svg>
                전체삭제
              </button>
            )}
          </div>

          {documents.length === 0 ? (
            <div className="text-center text-gray-500 dark:text-gray-400 py-8">
              <svg className="w-12 h-12 mx-auto mb-3 opacity-50" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
              <p className="text-sm">문서를 업로드하세요</p>
            </div>
          ) : (
            <div className="space-y-2">
              {documents.map(doc => (
                <div
                  key={doc.doc_id}
                  className="group p-3 bg-white/50 dark:bg-gray-700/50 rounded-xl border border-gray-200/50 dark:border-gray-600/50 hover:shadow-md transition-all"
                >
                  <div className="flex items-start justify-between">
                    <div className="flex-1 min-w-0 pr-2">
                      <p className="text-sm font-medium text-gray-800 dark:text-gray-200 truncate">
                        {doc.filename}
                      </p>
                      <div className="flex items-center space-x-2 mt-1">
                        <span className="px-2 py-0.5 text-xs bg-blue-100 dark:bg-blue-900/50 text-blue-600 dark:text-blue-400 rounded">
                          {doc.file_type.toUpperCase()}
                        </span>
                        <span className="text-xs text-gray-500 dark:text-gray-400">
                          {doc.total_chunks} 청크
                        </span>
                      </div>
                    </div>
                    {/* 개별 삭제 버튼 - 항상 보임 */}
                    <button
                      onClick={() => handleDeleteDocument(doc.doc_id)}
                      className="flex-shrink-0 p-1.5 text-gray-400 hover:text-red-500 hover:bg-red-100 dark:hover:bg-red-900/30 rounded-lg transition-all"
                      title={`${doc.filename} 삭제`}
                    >
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                      </svg>
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* 검색 설정 */}
        <div className="p-4 border-t border-gray-200/50 dark:border-gray-700/50">
          <button
            onClick={() => setShowSettings(!showSettings)}
            className="flex items-center justify-between w-full text-sm text-gray-600 dark:text-gray-400 hover:text-gray-800 dark:hover:text-gray-200"
          >
            <span className="flex items-center">
              <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
              </svg>
              검색 설정
            </span>
            <svg className={`w-4 h-4 transition-transform ${showSettings ? 'rotate-180' : ''}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
            </svg>
          </button>

          {showSettings && (
            <div className="mt-3 space-y-3">
              {/* 검색 방법 */}
              <div>
                <label className="text-xs text-gray-500 dark:text-gray-400">검색 방법</label>
                <select
                  value={searchMethod}
                  onChange={(e) => setSearchMethod(e.target.value as typeof searchMethod)}
                  className="mt-1 w-full px-3 py-2 text-sm bg-white dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-lg"
                >
                  <option value="hybrid">하이브리드 (권장)</option>
                  <option value="sparse">Sparse (BM25)</option>
                  <option value="dense">Dense (벡터)</option>
                </select>
              </div>

              {/* Alpha 슬라이더 */}
              {searchMethod === 'hybrid' && (
                <div>
                  <label className="text-xs text-gray-500 dark:text-gray-400">
                    Alpha: {alpha.toFixed(1)} (Dense 가중치)
                  </label>
                  <input
                    type="range"
                    min="0"
                    max="1"
                    step="0.1"
                    value={alpha}
                    onChange={(e) => setAlpha(parseFloat(e.target.value))}
                    className="mt-1 w-full"
                  />
                  <div className="flex justify-between text-xs text-gray-400">
                    <span>키워드</span>
                    <span>의미</span>
                  </div>
                </div>
              )}

              {/* Re-ranker 토글 */}
              <div className="flex items-center justify-between">
                <label className="text-xs text-gray-500 dark:text-gray-400">Re-ranking</label>
                <button
                  onClick={() => setUseReranker(!useReranker)}
                  className={`relative w-10 h-5 rounded-full transition-colors ${
                    useReranker ? 'bg-emerald-500' : 'bg-gray-300 dark:bg-gray-600'
                  }`}
                >
                  <span className={`absolute top-0.5 left-0.5 w-4 h-4 bg-white rounded-full transition-transform ${
                    useReranker ? 'translate-x-5' : ''
                  }`} />
                </button>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* 메인 채팅 영역 */}
      <div className="flex-1 flex flex-col">
        {/* 메시지 영역 */}
        <div className="flex-1 overflow-y-auto p-4 sm:p-6 space-y-4">
          {messages.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full text-center">
              <div className="relative mb-6">
                <div className="absolute inset-0 bg-gradient-to-r from-emerald-500 to-teal-500 rounded-full blur-2xl opacity-30 animate-pulse"></div>
                <div className="relative w-20 h-20 bg-gradient-to-r from-emerald-500 to-teal-500 rounded-2xl flex items-center justify-center shadow-2xl">
                  <svg className="w-12 h-12 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
                  </svg>
                </div>
              </div>
              <h2 className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-emerald-600 to-teal-600 dark:from-emerald-400 dark:to-teal-400 mb-3">
                RAG 문서 기반 채팅
              </h2>
              <p className="text-gray-600 dark:text-gray-400 mb-4 max-w-md">
                PDF, Markdown, JSON, TXT 파일을 업로드하고<br />
                하이브리드 검색으로 정확한 답변을 받아보세요
              </p>
              <div className="grid grid-cols-2 gap-3 max-w-md">
                <div className="p-3 bg-emerald-50 dark:bg-emerald-900/20 rounded-xl border border-emerald-200/50 dark:border-emerald-800/50">
                  <p className="text-xs font-medium text-emerald-700 dark:text-emerald-300">BM25 + Vector</p>
                  <p className="text-xs text-emerald-600 dark:text-emerald-400 mt-1">하이브리드 검색</p>
                </div>
                <div className="p-3 bg-teal-50 dark:bg-teal-900/20 rounded-xl border border-teal-200/50 dark:border-teal-800/50">
                  <p className="text-xs font-medium text-teal-700 dark:text-teal-300">Cross-Encoder</p>
                  <p className="text-xs text-teal-600 dark:text-teal-400 mt-1">Re-ranking</p>
                </div>
              </div>
            </div>
          ) : (
            <>
              {messages.map(message => (
                <div
                  key={message.id}
                  className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
                >
                  {message.role === 'system' ? (
                    <div className="w-full px-4 py-2 bg-gray-100 dark:bg-gray-700/50 rounded-xl text-sm text-gray-600 dark:text-gray-400">
                      <ReactMarkdown remarkPlugins={[remarkGfm]}>
                        {message.content}
                      </ReactMarkdown>
                    </div>
                  ) : message.role === 'user' ? (
                    <div className="max-w-[80%] px-4 py-3 bg-gradient-to-r from-emerald-500 to-teal-500 text-white rounded-2xl rounded-br-md shadow-lg">
                      <p className="text-sm">{message.content}</p>
                    </div>
                  ) : (
                    <div className="max-w-[85%] space-y-2">
                      <div className="px-4 py-3 bg-white dark:bg-gray-700 rounded-2xl rounded-bl-md shadow-lg border border-gray-200/50 dark:border-gray-600/50">
                        {/* 신뢰도 & 검색 방법 배지 */}
                        {(message.confidence || message.searchMethod) && (
                          <div className="flex items-center space-x-2 mb-2">
                            {message.confidence && (
                              <span className={`px-2 py-0.5 text-xs rounded-full ${getConfidenceColor(message.confidence)}`}>
                                신뢰도: {message.confidence}
                              </span>
                            )}
                            {message.searchMethod && (
                              <span className="px-2 py-0.5 text-xs rounded-full bg-blue-100 dark:bg-blue-900/30 text-blue-600 dark:text-blue-400">
                                {message.searchMethod}
                              </span>
                            )}
                          </div>
                        )}
                        
                        {/* 답변 내용 */}
                        <div className="prose prose-sm dark:prose-invert max-w-none">
                          <ReactMarkdown remarkPlugins={[remarkGfm]}>
                            {message.content}
                          </ReactMarkdown>
                        </div>
                      </div>

                      {/* 출처 */}
                      {message.sources && message.sources.length > 0 && (
                        <div className="px-3 py-2 bg-gray-50 dark:bg-gray-800/50 rounded-xl">
                          <p className="text-xs font-semibold text-gray-500 dark:text-gray-400 mb-2">
                            📚 출처 ({message.sources.length}개)
                          </p>
                          <div className="space-y-1">
                            {message.sources.slice(0, 3).map((source, idx) => (
                              <div key={idx} className="flex items-start space-x-2 text-xs">
                                <span className="flex-shrink-0 px-1.5 py-0.5 bg-emerald-100 dark:bg-emerald-900/30 text-emerald-600 dark:text-emerald-400 rounded">
                                  {source.rank}
                                </span>
                                <span className="text-gray-600 dark:text-gray-400 truncate">
                                  {source.filename} (점수: {source.score.toFixed(3)})
                                </span>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              ))}
              {isLoading && (
                <div className="flex items-start space-x-3">
                  <div className="flex items-center space-x-2 bg-gray-100 dark:bg-gray-700 px-4 py-3 rounded-2xl">
                    <div className="w-2 h-2 bg-emerald-500 rounded-full animate-bounce"></div>
                    <div className="w-2 h-2 bg-teal-500 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                    <div className="w-2 h-2 bg-cyan-500 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </>
          )}
        </div>

        {/* 입력 영역 */}
        <div className="border-t border-gray-200/50 dark:border-gray-700/50 p-4 bg-white/50 dark:bg-gray-800/50 backdrop-blur-sm">
          <div className="flex space-x-3">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && !e.shiftKey && handleSendMessage()}
              placeholder={documents.length > 0 ? "문서에 대해 질문하세요..." : "먼저 문서를 업로드하세요"}
              disabled={documents.length === 0 || isLoading}
              className="flex-1 px-4 py-3 bg-white dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-xl focus:outline-none focus:ring-2 focus:ring-emerald-500 disabled:opacity-50 disabled:cursor-not-allowed"
            />
            <button
              onClick={handleSendMessage}
              disabled={!input.trim() || isLoading || documents.length === 0}
              className="px-6 py-3 bg-gradient-to-r from-emerald-500 to-teal-500 text-white rounded-xl hover:from-emerald-600 hover:to-teal-600 disabled:opacity-50 disabled:cursor-not-allowed transition-all shadow-lg hover:shadow-xl"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
              </svg>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

