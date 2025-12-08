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

    try {
      // API 호출
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

      // AI 응답 추가
      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: data.answer || data.message || '응답 없음',
        timestamp: new Date(),
        toolsUsed: data.tools_used || [],
        frontmatterFile: data.frontmatter_file
      };

      setMessages(prev => [...prev, assistantMessage]);
      setUploadedFiles([]); // 파일 초기화

    } catch (error) {
      console.error('Error:', error);
      
      // 에러 메시지 추가
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: `오류가 발생했습니다: ${error instanceof Error ? error.message : '알 수 없는 오류'}`,
        timestamp: new Date()
      };

      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-[calc(100vh-12rem)] bg-white dark:bg-gray-800 rounded-2xl shadow-xl overflow-hidden">
      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto p-4 sm:p-6 space-y-4">
        {messages.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-center">
            <div className="text-6xl mb-4">💬</div>
            <h2 className="text-2xl font-semibold text-gray-700 dark:text-gray-300 mb-2">
              채팅을 시작해보세요
            </h2>
            <p className="text-gray-500 dark:text-gray-400">
              일반 대화를 하거나 HTML 파일을 업로드하여 분석할 수 있습니다.
            </p>
          </div>
        ) : (
          <>
            {messages.map(message => (
              <ChatMessage key={message.id} message={message} />
            ))}
            {isLoading && (
              <div className="flex items-center space-x-2 text-gray-500">
                <div className="animate-bounce">●</div>
                <div className="animate-bounce delay-100">●</div>
                <div className="animate-bounce delay-200">●</div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </>
        )}
      </div>

      {/* File Upload Area */}
      {uploadedFiles.length > 0 && (
        <div className="px-4 py-2 bg-gray-50 dark:bg-gray-700 border-t border-gray-200 dark:border-gray-600">
          <FileUpload 
            files={uploadedFiles} 
            onFilesChange={setUploadedFiles}
          />
        </div>
      )}

      {/* Input Area */}
      <div className="border-t border-gray-200 dark:border-gray-700 p-4 bg-gray-50 dark:bg-gray-700">
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

