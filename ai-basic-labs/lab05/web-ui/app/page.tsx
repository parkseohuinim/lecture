'use client';

import { useState, useRef, useEffect } from 'react';
import { ChatInterface } from '@/components/ChatInterface';

export default function Home() {
  return (
    <main className="flex min-h-screen flex-col items-center bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-900 dark:to-gray-800">
      <div className="w-full max-w-5xl p-4 sm:p-6 lg:p-8">
        {/* Header */}
        <div className="mb-8 text-center">
          <h1 className="text-4xl font-bold text-gray-900 dark:text-white mb-2">
            MCP Lab05 Chat
          </h1>
          <p className="text-gray-600 dark:text-gray-400">
            AI 챗봇과 대화하거나 HTML 파일을 분석해보세요
          </p>
        </div>

        {/* Chat Interface */}
        <ChatInterface />
      </div>
    </main>
  );
}
