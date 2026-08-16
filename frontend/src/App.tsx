import React, { useState } from 'react';
import { Scale, Search, ShieldCheck, FileText, AlertTriangle, Network, Upload } from 'lucide-react';
import DocumentUpload from './components/DocumentUpload';
import GraphVisualizer from './components/GraphVisualizer';

interface Claim {
  text: string;
  evidence_ids: string[];
  support_status?: string;
}

interface Source {
  evidence_id: string;
  title?: string;
  case_name?: string;
  court?: string;
  page?: number;
  paragraph?: number;
  url?: string;
  section?: string;
  authority_level: number;
}

interface Confidence {
  label: string;
  reasons: string[];
}

interface AnswerData {
  answer: string;
  claims: Claim[];
  sources: Source[];
  uncertainties: string[];
  confidence: Confidence;
  disclaimer: string;
}

export default function App() {
  const [activeTab, setActiveTab] = useState<'research' | 'upload' | 'graph'>('research');
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [data, setData] = useState<AnswerData | null>(null);
  const [selectedEvidence, setSelectedEvidence] = useState<Source | null>(null);

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;

    setLoading(true);
    setData(null);
    setSelectedEvidence(null);

    try {
      const res = await fetch('/answer', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: query }),
      });
      if (!res.ok) throw new Error('API error');
      const result: AnswerData = await res.json();
      setData(result);
    } catch (err) {
      console.error(err);
      alert('Error fetching answer from backend. Ensure API is running.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      {/* Navbar */}
      <header className="navbar">
        <div className="nav-brand">
          <Scale size={26} />
          <span>LegalGuard</span>
        </div>

        {/* Top Navigation Tabs */}
        <nav className="nav-tabs">
          <button
            className={`nav-tab ${activeTab === 'research' ? 'active' : ''}`}
            onClick={() => setActiveTab('research')}
          >
            <Search size={16} />
            Research & Q&A
          </button>

          <button
            className={`nav-tab ${activeTab === 'upload' ? 'active' : ''}`}
            onClick={() => setActiveTab('upload')}
          >
            <Upload size={16} />
            Ingest Document
          </button>

          <button
            className={`nav-tab ${activeTab === 'graph' ? 'active' : ''}`}
            onClick={() => setActiveTab('graph')}
          >
            <Network size={16} />
            Knowledge Graph
          </button>
        </nav>

        <div className="nav-badge">Neo4j GraphRAG + Qwen2.5-3B</div>
      </header>

      {/* Main Content */}
      <main className="main-content">
        {/* Tab 1: Research & Q&A */}
        {activeTab === 'research' && (
          <>
            <section className="search-section">
              <h1 className="search-title">Explainable Indian Law Research Assistant</h1>
              <p className="search-subtitle">
                Grounded in Neo4j Legal Knowledge Graph with Deterministic Citations & Provenance
              </p>

              <form onSubmit={handleSearch} className="search-box">
                <input
                  type="text"
                  className="search-input"
                  placeholder="Ask a question (e.g. What does Article 21 protect and which cases interpreted it?)"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                />
                <button type="submit" className="search-button" disabled={loading}>
                  {loading ? 'Searching...' : 'Search'}
                </button>
              </form>
            </section>

            {/* Results Grid */}
            {data && (
              <div className="results-grid">
                {/* Left Column: Answer & Claims */}
                <div className="left-col">
                  <div className="card">
                    <div className="card-title">
                      <FileText size={20} />
                      Legal Answer
                    </div>
                    <p style={{ whiteSpace: 'pre-line', marginBottom: '1rem' }}>{data.answer}</p>
                  </div>

                  {/* Claims Breakdown */}
                  {data.claims.length > 0 && (
                    <div className="card">
                      <div className="card-title">
                        <ShieldCheck size={20} />
                        Verified Claims & Citation Mapping
                      </div>
                      {data.claims.map((claim, idx) => (
                        <div key={idx} style={{ marginBottom: '0.75rem' }}>
                          <span style={{ fontWeight: 500 }}>• {claim.text}</span>
                          {claim.evidence_ids.map((eid) => (
                            <button
                              key={eid}
                              className="citation-badge"
                              onClick={() => {
                                const src = data.sources.find((s) => s.evidence_id === eid);
                                if (src) setSelectedEvidence(src);
                              }}
                            >
                              [{eid}]
                            </button>
                          ))}
                        </div>
                      ))}
                    </div>
                  )}

                  {/* Uncertainties */}
                  {data.uncertainties.length > 0 && (
                    <div className="card" style={{ borderColor: 'rgba(245,158,11,0.4)' }}>
                      <div className="card-title" style={{ color: 'var(--accent-gold)' }}>
                        <AlertTriangle size={20} />
                        Limitations & Uncertainties
                      </div>
                      <ul>
                        {data.uncertainties.map((u, idx) => (
                          <li key={idx} style={{ marginLeft: '1.25rem', color: 'var(--text-muted)' }}>
                            {u}
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>

                {/* Right Column: Sources & Confidence */}
                <div className="right-col">
                  {/* Confidence Card */}
                  <div className="card">
                    <div className="card-title">
                      <ShieldCheck size={20} />
                      Evidence Confidence
                    </div>
                    <div
                      style={{
                        fontSize: '1.25rem',
                        fontWeight: 700,
                        textTransform: 'uppercase',
                        color:
                          data.confidence.label === 'high'
                            ? 'var(--accent-green)'
                            : data.confidence.label === 'medium'
                            ? 'var(--accent-gold)'
                            : 'var(--accent-red)',
                        marginBottom: '0.5rem',
                      }}
                    >
                      {data.confidence.label} Confidence
                    </div>
                    <ul style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginLeft: '1rem' }}>
                      {data.confidence.reasons.map((reason, idx) => (
                        <li key={idx}>{reason}</li>
                      ))}
                    </ul>
                  </div>

                  {/* Sources Drawer */}
                  <div className="card">
                    <div className="card-title">
                      <Network size={20} />
                      Retrieved Authorities ({data.sources.length})
                    </div>

                    {data.sources.map((source) => (
                      <div
                        key={source.evidence_id}
                        className="evidence-card"
                        style={{
                          borderLeftColor:
                            selectedEvidence?.evidence_id === source.evidence_id
                              ? 'var(--accent-blue)'
                              : 'var(--accent-gold)',
                        }}
                      >
                        <div className="evidence-header">
                          [{source.evidence_id}] {source.title || source.case_name || 'Legal Document'}
                        </div>
                        {source.court && (
                          <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                            Court: {source.court}
                          </div>
                        )}
                        {source.section && (
                          <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                            Provision: {source.section}
                          </div>
                        )}
                        {source.url && (
                          <a
                            href={source.url}
                            target="_blank"
                            rel="noreferrer"
                            className="evidence-url"
                          >
                            View Official Source ↗
                          </a>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}
          </>
        )}

        {/* Tab 2: Document Upload */}
        {activeTab === 'upload' && <DocumentUpload />}

        {/* Tab 3: Knowledge Graph Visualizer */}
        {activeTab === 'graph' && <GraphVisualizer />}

        {/* Legal Disclaimer */}
        <div className="disclaimer-banner">
          ⚖️ <strong>Disclaimer:</strong> This system provides AI-assisted legal information and
          research support based on retrieved sources. It is not a lawyer and does not provide
          formal legal advice. Always verify claims against authoritative original sources.
        </div>
      </main>
    </div>
  );
}
