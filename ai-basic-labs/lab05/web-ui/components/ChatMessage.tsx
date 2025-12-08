'use client';

import { Message } from './ChatInterface';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

interface ChatMessageProps {
  message: Message;
}

export function ChatMessage({ message }: ChatMessageProps) {
  const isUser = message.role === 'user';

  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div
        className={`max-w-[80%] rounded-2xl px-4 py-3 ${
          isUser
            ? 'bg-blue-500 text-white'
            : 'bg-gray-100 dark:bg-gray-700 text-gray-900 dark:text-white'
        }`}
      >
        {/* 메시지 내용 - AI 응답은 항상 Markdown 렌더링 */}
        {!isUser ? (
          <div className="prose prose-sm dark:prose-invert max-w-none markdown-content">
            <ReactMarkdown
              remarkPlugins={[remarkGfm]}
              components={{
                // 테이블 스타일링
                table: ({ node, ...props }) => (
                  <div className="overflow-x-auto my-4">
                    <table className="min-w-full border border-gray-300 dark:border-gray-600" {...props} />
                  </div>
                ),
                th: ({ node, ...props }) => (
                  <th className="px-3 py-2 text-left text-xs font-semibold bg-gray-200 dark:bg-gray-600 border border-gray-300 dark:border-gray-500" {...props} />
                ),
                td: ({ node, ...props }) => (
                  <td className="px-3 py-2 text-sm border border-gray-300 dark:border-gray-600" {...props} />
                ),
                // 코드 블록
                code: ({ node, inline, ...props }: any) => (
                  inline ? (
                    <code className="px-1.5 py-0.5 rounded bg-gray-200 dark:bg-gray-600 font-mono text-sm" {...props} />
                  ) : (
                    <code className="block p-3 rounded-lg bg-gray-900 text-gray-100 overflow-x-auto font-mono text-sm my-2" {...props} />
                  )
                ),
                // 헤더
                h1: ({ node, ...props }) => <h1 className="text-2xl font-bold mt-4 mb-2 text-gray-900 dark:text-white" {...props} />,
                h2: ({ node, ...props }) => <h2 className="text-xl font-bold mt-3 mb-2 text-gray-900 dark:text-white" {...props} />,
                h3: ({ node, ...props }) => <h3 className="text-lg font-semibold mt-2 mb-1 text-gray-900 dark:text-white" {...props} />,
                // 리스트
                ul: ({ node, ...props }) => <ul className="list-disc list-inside my-2 space-y-1" {...props} />,
                ol: ({ node, ...props }) => <ol className="list-decimal list-inside my-2 space-y-1" {...props} />,
                // 단락
                p: ({ node, ...props }) => <p className="my-2" {...props} />,
              }}
            >
              {message.content}
            </ReactMarkdown>
          </div>
        ) : (
          <div className="text-sm sm:text-base whitespace-pre-wrap break-words">
            {message.content}
          </div>
        )}

        {/* 파일 정보 */}
        {message.files && message.files.length > 0 && (
          <div className="mt-2 pt-2 border-t border-white/20 space-y-1">
            {message.files.map((file, idx) => (
              <div key={idx} className="text-xs opacity-80 flex items-center space-x-2">
                <span>📎</span>
                <span>{file.name}</span>
                <span className="text-xs">({(file.size / 1024).toFixed(1)} KB)</span>
              </div>
            ))}
          </div>
        )}

        {/* 사용된 도구 */}
        {message.toolsUsed && message.toolsUsed.length > 0 && (
          <div className="mt-2 pt-2 border-t border-gray-300 dark:border-gray-600">
            <div className="text-xs opacity-70 flex items-center space-x-2">
              <span>🔧</span>
              <span>도구: {message.toolsUsed.join(', ')}</span>
            </div>
          </div>
        )}

        {/* Frontmatter 파일 다운로드 */}
        {message.frontmatterFile && (
          <div className="mt-2 pt-2 border-t border-gray-300 dark:border-gray-600">
            <button
              onClick={async () => {
                try {
                  const response = await fetch(`http://localhost:8000${message.frontmatterFile!.download_url}`);
                  const blob = await response.blob();
                  const url = window.URL.createObjectURL(blob);
                  const a = document.createElement('a');
                  a.href = url;
                  a.download = message.frontmatterFile!.filename;
                  document.body.appendChild(a);
                  a.click();
                  window.URL.revokeObjectURL(url);
                  document.body.removeChild(a);
                } catch (error) {
                  console.error('다운로드 실패:', error);
                  alert('파일 다운로드에 실패했습니다.');
                }
              }}
              className="text-xs flex items-center space-x-2 hover:underline cursor-pointer bg-transparent border-none p-0"
            >
              <span>📥</span>
              <span>Frontmatter 파일 다운로드</span>
              <span className="opacity-70">({(message.frontmatterFile.size / 1024).toFixed(1)} KB)</span>
            </button>
          </div>
        )}

        {/* 타임스탬프 */}
        <div className={`text-xs mt-1 ${isUser ? 'text-blue-100' : 'text-gray-500 dark:text-gray-400'}`}>
          {message.timestamp.toLocaleTimeString('ko-KR', {
            hour: '2-digit',
            minute: '2-digit'
          })}
        </div>
      </div>
    </div>
  );
}

