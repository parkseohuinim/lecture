'use client';

import { useState, useRef, useEffect } from 'react';
import { ChatInterface } from '@/components/ChatInterface';
import { ConferenceRoom } from '@/components/ConferenceRoom';
import { RAGChat } from '@/components/RAGChat';

type Tab = 'chat' | 'rag' | 'conference';

export default function Home() {
  const [activeTab, setActiveTab] = useState<Tab>('chat');

  return (
    <main className="flex min-h-screen flex-col items-center bg-gradient-to-br from-indigo-50 via-purple-50 to-pink-50 dark:from-gray-900 dark:via-purple-900/20 dark:to-indigo-900/20 relative overflow-hidden">
      {/* Animated background elements */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute -top-1/2 -left-1/2 w-full h-full bg-gradient-to-br from-blue-400/10 to-purple-400/10 rounded-full blur-3xl animate-pulse-slow"></div>
        <div className="absolute -bottom-1/2 -right-1/2 w-full h-full bg-gradient-to-tl from-pink-400/10 to-indigo-400/10 rounded-full blur-3xl animate-pulse-slower"></div>
      </div>

      <div className="w-full max-w-6xl p-4 sm:p-6 lg:p-8 relative z-10">
        {/* Header with glass effect */}
        <div className="mb-8 text-center">
          <div className="inline-flex items-center justify-center space-x-3 mb-4">
            <div className="relative">
              <div className="absolute inset-0 bg-gradient-to-r from-blue-500 to-purple-500 rounded-2xl blur-xl opacity-50 animate-pulse-slow"></div>
              <div className="relative bg-gradient-to-r from-blue-500 to-purple-500 p-3 rounded-2xl">
                <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                </svg>
              </div>
            </div>
            <h1 className="text-5xl font-extrabold bg-clip-text text-transparent bg-gradient-to-r from-blue-600 via-purple-600 to-pink-600 dark:from-blue-400 dark:via-purple-400 dark:to-pink-400">
              Intelligent Agent
            </h1>
          </div>
          <p className="text-lg text-gray-700 dark:text-gray-300 font-medium mb-2">
            Smart Document Processing Assistant
          </p>
          <p className="text-gray-600 dark:text-gray-400 max-w-2xl mx-auto">
            지능형 문서 분석 • HTML 자동 파싱 • 스마트 컨텍스트 추출
          </p>
          
          {/* Feature badges */}
          <div className="flex flex-wrap justify-center gap-2 mt-4">
            <span className="px-3 py-1 text-xs font-semibold bg-white/60 dark:bg-gray-800/60 backdrop-blur-sm rounded-full text-blue-600 dark:text-blue-400 border border-blue-200 dark:border-blue-800">
              AI-Powered
            </span>
            <span className="px-3 py-1 text-xs font-semibold bg-white/60 dark:bg-gray-800/60 backdrop-blur-sm rounded-full text-purple-600 dark:text-purple-400 border border-purple-200 dark:border-purple-800">
              Real-time Processing
            </span>
            <span className="px-3 py-1 text-xs font-semibold bg-white/60 dark:bg-gray-800/60 backdrop-blur-sm rounded-full text-pink-600 dark:text-pink-400 border border-pink-200 dark:border-pink-800">
              Smart Document Analysis
            </span>
          </div>
        </div>

        {/* Tabs */}
        <div className="mb-6 flex justify-center">
          <div className="glass rounded-2xl p-1 inline-flex space-x-1">
            <button
              onClick={() => setActiveTab('chat')}
              className={`px-5 py-2.5 rounded-xl font-medium transition-all duration-200 cursor-pointer ${
                activeTab === 'chat'
                  ? 'bg-gradient-to-r from-blue-500 to-purple-500 text-white shadow-lg'
                  : 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-200'
              }`}
            >
              <div className="flex items-center space-x-2">
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
                </svg>
                <span>채팅</span>
              </div>
            </button>
            <button
              onClick={() => setActiveTab('rag')}
              className={`px-5 py-2.5 rounded-xl font-medium transition-all duration-200 cursor-pointer ${
                activeTab === 'rag'
                  ? 'bg-gradient-to-r from-emerald-500 to-teal-500 text-white shadow-lg'
                  : 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-200'
              }`}
            >
              <div className="flex items-center space-x-2">
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
                </svg>
                <span>RAG</span>
              </div>
            </button>
            <button
              onClick={() => setActiveTab('conference')}
              className={`px-5 py-2.5 rounded-xl font-medium transition-all duration-200 cursor-pointer ${
                activeTab === 'conference'
                  ? 'bg-gradient-to-r from-blue-500 to-purple-500 text-white shadow-lg'
                  : 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-200'
              }`}
            >
              <div className="flex items-center space-x-2">
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z" />
                </svg>
                <span>회의실</span>
              </div>
            </button>
          </div>
        </div>

        {/* Content */}
        {activeTab === 'chat' && <ChatInterface />}
        {activeTab === 'rag' && <RAGChat />}
        {activeTab === 'conference' && <ConferenceRoom />}
      </div>
    </main>
  );
}
