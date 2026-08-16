import React, { useState } from 'react';
import { Upload, FileText, CheckCircle, AlertCircle, Loader2 } from 'lucide-react';

interface IngestResult {
  status: string;
  document_id?: string;
  chunks_created?: number;
  entities_extracted?: number;
  relationships_created?: number;
  message?: string;
  dry_run?: boolean;
}

export default function DocumentUpload() {
  const [file, setFile] = useState<File | null>(null);
  const [documentType, setDocumentType] = useState('ACT');
  const [title, setTitle] = useState('');
  const [authorityLevel, setAuthorityLevel] = useState(5);
  const [dryRun, setDryRun] = useState(false);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<IngestResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [dragActive, setDragActive] = useState(false);

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      setFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file) {
      setError('Please select a file to upload.');
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    const formData = new FormData();
    formData.append('file', file);
    formData.append('document_type', documentType);
    if (title.trim()) formData.append('title', title.trim());
    formData.append('authority_level', authorityLevel.toString());
    formData.append('dry_run', dryRun ? 'true' : 'false');

    try {
      const res = await fetch('/ingest/upload', {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({ detail: 'Upload failed' }));
        throw new Error(errData.detail || 'Upload ingestion failed');
      }

      const data: IngestResult = await res.json();
      setResult(data);
    } catch (err: any) {
      console.error(err);
      setError(err.message || 'An unexpected error occurred during document upload.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="upload-container">
      <div className="card">
        <div className="card-title">
          <Upload size={22} />
          Automated Legal Document Ingestion
        </div>
        <p className="search-subtitle" style={{ textAlign: 'left', marginBottom: '1.5rem' }}>
          Upload PDFs, Markdown, or Text files to automatically extract legal entities, section hierarchies, and vector embeddings into Neo4j.
        </p>

        <form onSubmit={handleSubmit}>
          {/* Dropzone */}
          <div
            className={`dropzone ${dragActive ? 'dropzone-active' : ''} ${file ? 'dropzone-selected' : ''}`}
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
          >
            <input
              type="file"
              id="file-upload"
              accept=".pdf,.txt,.md"
              onChange={handleFileChange}
              style={{ display: 'none' }}
            />

            {file ? (
              <div className="file-info">
                <FileText size={36} className="file-icon" />
                <div>
                  <div className="file-name">{file.name}</div>
                  <div className="file-size">{(file.size / 1024).toFixed(1)} KB • {file.type || 'Document'}</div>
                </div>
                <button
                  type="button"
                  className="change-file-btn"
                  onClick={() => setFile(null)}
                >
                  Change File
                </button>
              </div>
            ) : (
              <label htmlFor="file-upload" className="dropzone-label">
                <Upload size={40} className="upload-icon" />
                <span className="dropzone-text">
                  Drag & Drop legal PDF/TXT file here, or <span className="browse-link">Browse Files</span>
                </span>
                <span className="dropzone-hint">Supports PDF, TXT, MD documents</span>
              </label>
            )}
          </div>

          {/* Configuration Form */}
          <div className="upload-form-grid">
            <div className="form-group">
              <label htmlFor="documentType" className="form-label">Document Type</label>
              <select
                id="documentType"
                className="form-input"
                value={documentType}
                onChange={(e) => setDocumentType(e.target.value)}
              >
                <option value="ACT">Act (Statute / Central Act)</option>
                <option value="CONSTITUTION">Constitution Provision</option>
                <option value="JUDGMENT">Court Judgment / Precedent</option>
                <option value="NOTIFICATION">Official Gazette Notification</option>
                <option value="CIRCULAR">Government Circular</option>
              </select>
            </div>

            <div className="form-group">
              <label htmlFor="title" className="form-label">Custom Document Title (Optional)</label>
              <input
                type="text"
                id="title"
                className="form-input"
                placeholder="e.g. The Bharatiya Nyaya Sanhita 2023"
                value={title}
                onChange={(e) => setTitle(e.target.value)}
              />
            </div>

            <div className="form-group">
              <label htmlFor="authorityLevel" className="form-label">Authority Level (0 - 5)</label>
              <input
                type="number"
                id="authorityLevel"
                min="0"
                max="5"
                className="form-input"
                value={authorityLevel}
                onChange={(e) => setAuthorityLevel(parseInt(e.target.value, 10) || 0)}
              />
            </div>

            <div className="form-group" style={{ display: 'flex', alignItems: 'center', marginTop: '1.5rem' }}>
              <label className="checkbox-label">
                <input
                  type="checkbox"
                  checked={dryRun}
                  onChange={(e) => setDryRun(e.target.checked)}
                />
                Dry Run (Simulate parsing without writing to graph)
              </label>
            </div>
          </div>

          <button
            type="submit"
            className="search-button upload-submit-btn"
            disabled={loading || !file}
          >
            {loading ? (
              <>
                <Loader2 size={18} className="spinner" /> Processing Ingestion Pipeline...
              </>
            ) : (
              'Upload & Ingest Document'
            )}
          </button>
        </form>
      </div>

      {/* Error Message */}
      {error && (
        <div className="card" style={{ borderColor: 'var(--accent-red)' }}>
          <div className="card-title" style={{ color: 'var(--accent-red)' }}>
            <AlertCircle size={20} /> Ingestion Error
          </div>
          <p>{error}</p>
        </div>
      )}

      {/* Success Result */}
      {result && (
        <div className="card" style={{ borderColor: 'var(--accent-green)' }}>
          <div className="card-title" style={{ color: 'var(--accent-green)' }}>
            <CheckCircle size={22} /> Document Successfully Ingested!
          </div>

          <div className="result-stats-grid">
            <div className="stat-box">
              <div className="stat-value">{result.document_id || 'OK'}</div>
              <div className="stat-label">Document ID</div>
            </div>
            <div className="stat-box">
              <div className="stat-value">{result.chunks_created ?? 'N/A'}</div>
              <div className="stat-label">Text Chunks Created</div>
            </div>
            <div className="stat-box">
              <div className="stat-value">{result.entities_extracted ?? 'N/A'}</div>
              <div className="stat-label">Entities Extracted</div>
            </div>
            <div className="stat-box">
              <div className="stat-value">{result.relationships_created ?? 'N/A'}</div>
              <div className="stat-label">Relationships Merged</div>
            </div>
          </div>

          {result.dry_run && (
            <p className="dry-run-note">
              ℹ️ Simulated Ingestion: No persistent changes were written to Neo4j.
            </p>
          )}
        </div>
      )}
    </div>
  );
}
