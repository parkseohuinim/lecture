'use client';

interface FileUploadProps {
  files: File[];
  onFilesChange: (files: File[]) => void;
}

export function FileUpload({ files, onFilesChange }: FileUploadProps) {
  const removeFile = (index: number) => {
    const newFiles = [...files];
    newFiles.splice(index, 1);
    onFilesChange(newFiles);
  };

  return (
    <div className="space-y-2">
      <div className="text-sm text-gray-600 dark:text-gray-400">업로드된 파일:</div>
      <div className="flex flex-wrap gap-2">
        {files.map((file, index) => (
          <div
            key={index}
            className="flex items-center space-x-2 bg-white dark:bg-gray-800 px-3 py-2 rounded-lg border border-gray-200 dark:border-gray-600"
          >
            <span className="text-sm text-gray-700 dark:text-gray-300">
              📄 {file.name}
            </span>
            <span className="text-xs text-gray-500">
              ({(file.size / 1024).toFixed(1)} KB)
            </span>
            <button
              onClick={() => removeFile(index)}
              className="text-red-500 hover:text-red-700 ml-2"
            >
              ✕
            </button>
          </div>
        ))}
      </div>
    </div>
  );
}

