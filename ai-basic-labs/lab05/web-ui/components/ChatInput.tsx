'use client';

import { useState, useRef } from 'react';

interface ChatInputProps {
  onSend: (message: string) => void;
  isLoading: boolean;
  onFileSelect: (files: File[]) => void;
  hasFiles: boolean;
}

export function ChatInput({ onSend, isLoading, onFileSelect, hasFiles }: ChatInputProps) {
  const [input, setInput] = useState('');
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (input.trim() || hasFiles) {
      onSend(input);
      setInput('');
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || []);
    onFileSelect(files);
  };

  return (
    <form onSubmit={handleSubmit} className="flex items-end space-x-2">
      {/* 파일 업로드 버튼 */}
      <button
        type="button"
        onClick={() => fileInputRef.current?.click()}
        disabled={isLoading}
        className="p-3 rounded-xl bg-gray-200 hover:bg-gray-300 dark:bg-gray-600 dark:hover:bg-gray-500 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
        title="HTML 파일 업로드"
      >
        <svg
          xmlns="http://www.w3.org/2000/svg"
          className="h-5 w-5 text-gray-700 dark:text-gray-200"
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M15.172 7l-6.586 6.586a2 2 0 102.828 2.828l6.414-6.586a4 4 0 00-5.656-5.656l-6.415 6.585a6 6 0 108.486 8.486L20.5 13"
          />
        </svg>
      </button>
      <input
        type="file"
        ref={fileInputRef}
        onChange={handleFileChange}
        accept=".html"
        multiple
        className="hidden"
      />

      {/* 메시지 입력 */}
      <div className="flex-1">
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={hasFiles ? "파일에 대해 질문하세요..." : "메시지를 입력하세요..."}
          disabled={isLoading}
          rows={1}
          className="w-full px-4 py-3 rounded-xl border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 text-gray-900 dark:text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-blue-500 resize-none disabled:opacity-50"
        />
      </div>

      {/* 전송 버튼 */}
      <button
        type="submit"
        disabled={isLoading || (!input.trim() && !hasFiles)}
        className="p-3 rounded-xl bg-blue-500 hover:bg-blue-600 text-white transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
      >
        <svg
          xmlns="http://www.w3.org/2000/svg"
          className="h-5 w-5"
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"
          />
        </svg>
      </button>
    </form>
  );
}

