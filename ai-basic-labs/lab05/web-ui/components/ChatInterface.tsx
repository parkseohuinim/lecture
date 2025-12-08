'use client';

import { useState, useRef, useEffect } from 'react';
import { ChatMessage } from './ChatMessage';
import { ChatInput } from './ChatInput';
import { FileUpload } from './FileUpload';

export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  files?: Array<{ name: string; size: number }>;
  toolsUsed?: string[];
  frontmatterFile?: {
    filename: string;
    download_url: string;
    size: number;
  };
}

export function ChatInterface() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [uploadedFiles, setUploadedFiles] = useState<File[]>([]);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSendMessage = async (content: string) => {
    if (!content.trim() && uploadedFiles.length === 0) return;

    // 사용자 메시지 추가
    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: content.trim() || '(파일 업로드)',
      timestamp: new Date(),
      files: uploadedFiles.map(f => ({ name: f.name, size: f.size }))
    };

    setMessages(prev => [...prev, userMessage]);
    setIsLoading(true);

    const assistantMessageId = (Date.now() + 1).toString();

    try {
      // 파일이 있으면 기존 방식 (MCP 도구 호출 필요)
      if (uploadedFiles.length > 0) {
        const formData = new FormData();
        formData.append('question', content);
        
        uploadedFiles.forEach(file => {
          formData.append('files', file);
        });

        const response = await fetch('http://localhost:8000/api/chat', {
          method: 'POST',
          body: formData
        });

        if (!response.ok) {
          throw new Error(`API 오류: ${response.statusText}`);
        }

        const data = await response.json();

        const assistantMessage: Message = {
          id: assistantMessageId,
          role: 'assistant',
          content: data.answer || data.message || '응답 없음',
          timestamp: new Date(),
          toolsUsed: data.tools_used || [],
          frontmatterFile: data.frontmatter_file
        };

        setMessages(prev => [...prev, assistantMessage]);
        setUploadedFiles([]);
      } else {
        // 파일이 없으면 스트리밍 방식 (ChatGPT 스타일)
        let streamedContent = '';

        // 빈 어시스턴트 메시지 먼저 추가
        setMessages(prev => [...prev, {
          id: assistantMessageId,
          role: 'assistant',
          content: '',
          timestamp: new Date()
        }]);

        const formData = new FormData();
        formData.append('question', content);

        const response = await fetch('http://localhost:8000/api/chat/stream', {
          method: 'POST',
          body: formData
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

                if (data.type === 'token') {
                  // 토큰 수신 - 타자치듯 추가
                  streamedContent += data.data;
                  
                  setMessages(prev => prev.map(msg => 
                    msg.id === assistantMessageId
                      ? { ...msg, content: streamedContent }
                      : msg
                  ));
                } else if (data.type === 'error') {
                  throw new Error(data.data);
                }
              } catch (parseError) {
                // JSON 파싱 실패 시 무시
              }
            }
          }
        }
      }

    } catch (error) {
      console.error('Error:', error);
      
      // 에러 메시지 처리
      setMessages(prev => {
        const hasEmptyAssistant = prev.some(msg => msg.id === assistantMessageId);
        if (hasEmptyAssistant) {
          return prev.map(msg => 
            msg.id === assistantMessageId
              ? { ...msg, content: `오류가 발생했습니다: ${error instanceof Error ? error.message : '알 수 없는 오류'}` }
              : msg
          );
        } else {
          return [...prev, {
            id: assistantMessageId,
            role: 'assistant' as const,
            content: `오류가 발생했습니다: ${error instanceof Error ? error.message : '알 수 없는 오류'}`,
            timestamp: new Date()
          }];
        }
      });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-[calc(100vh-16rem)] glass rounded-3xl shadow-2xl overflow-hidden border border-white/20 dark:border-gray-700/50">
      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto p-4 sm:p-6 space-y-4">
        {messages.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-center">
            <div className="relative mb-6">
              <div className="absolute inset-0 bg-gradient-to-r from-blue-500 to-purple-500 rounded-full blur-2xl opacity-30 animate-pulse-slow"></div>
              <div className="relative w-20 h-20 bg-gradient-to-r from-blue-500 to-purple-500 rounded-2xl flex items-center justify-center animate-float shadow-2xl">
                <svg className="w-12 h-12 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                </svg>
              </div>
            </div>
            <h2 className="text-3xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-600 to-purple-600 dark:from-blue-400 dark:to-purple-400 mb-3">
              Agent와 대화를 시작하세요
            </h2>
            <p className="text-gray-600 dark:text-gray-400 mb-6 max-w-md">
              일반 대화를 하거나 HTML 파일을 업로드하여 지능형 문서 분석을 시작할 수 있습니다
            </p>
            
            {/* Example prompts */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 mt-4 w-full max-w-2xl">
              <div className="p-4 bg-gradient-to-br from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 rounded-xl border border-blue-200/50 dark:border-blue-800/50">
                <svg className="w-6 h-6 text-blue-500 mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
                </svg>
                <p className="text-sm font-medium text-gray-700 dark:text-gray-300">일반 대화</p>
                <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">"HTML이란 무엇인가요?"</p>
              </div>
              <div className="p-4 bg-gradient-to-br from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20 rounded-xl border border-purple-200/50 dark:border-purple-800/50">
                <svg className="w-6 h-6 text-purple-500 mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
                <p className="text-sm font-medium text-gray-700 dark:text-gray-300">문서 분석</p>
                <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">"이 파일의 내용을 추출해줘"</p>
              </div>
            </div>
          </div>
        ) : (
          <>
            {messages.map(message => (
              <ChatMessage key={message.id} message={message} />
            ))}
            {isLoading && (
              <div className="flex items-start space-x-3">
                <div className="flex-shrink-0 w-8 h-8 rounded-full bg-gradient-to-r from-blue-500 to-purple-500 flex items-center justify-center">
                  <svg className="w-5 h-5 text-white animate-pulse" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                  </svg>
                </div>
                <div className="flex items-center space-x-2 bg-gray-100 dark:bg-gray-700 px-4 py-3 rounded-2xl">
                  <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce"></div>
                  <div className="w-2 h-2 bg-purple-500 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                  <div className="w-2 h-2 bg-pink-500 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </>
        )}
      </div>

      {/* File Upload Area */}
      {uploadedFiles.length > 0 && (
        <div className="px-4 py-3 bg-gradient-to-r from-blue-50/50 to-purple-50/50 dark:from-blue-900/10 dark:to-purple-900/10 border-t border-blue-200/50 dark:border-blue-800/50 backdrop-blur-sm">
          <FileUpload 
            files={uploadedFiles} 
            onFilesChange={setUploadedFiles}
          />
        </div>
      )}

      {/* Input Area */}
      <div className="border-t border-gray-200/50 dark:border-gray-700/50 p-4 bg-white/50 dark:bg-gray-800/50 backdrop-blur-sm">
        <ChatInput
          onSend={handleSendMessage}
          isLoading={isLoading}
          onFileSelect={setUploadedFiles}
          hasFiles={uploadedFiles.length > 0}
        />
      </div>
    </div>
  );
}

