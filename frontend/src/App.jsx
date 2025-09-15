import { useEffect, useState } from "react";

// Secure API base configuration
const API_BASE = import.meta.env.VITE_API_BASE;
if (!API_BASE) {
  throw new Error("VITE_API_BASE environment variable is required");
}

export default function App() {
  const [health, setHealth] = useState(null);
  const [recent, setRecent] = useState([]);
  const [postResult, setPostResult] = useState(null);
  const [form, setForm] = useState({
    external_id: "",
    customer_id: "",
    source: "chatbot",
    raw_text: "",
    sentiment: "",
  });
  const [error, setError] = useState("");

  // NEW: selection + summary UI state
  const [selectedConv, setSelectedConv] = useState(null);
  const [summary, setSummary] = useState("");
  const [similar, setSimilar] = useState([]);
  const [loadingSummary, setLoadingSummary] = useState(false);
  const [uiError, setUiError] = useState("");

  // Input validation functions
  const validateForm = (formData) => {
    const errors = [];
    
    // Validate external_id
    if (formData.external_id && (formData.external_id.length > 100 || !/^[a-zA-Z0-9_-]*$/.test(formData.external_id))) {
      errors.push("External ID must be alphanumeric with dashes/underscores and under 100 characters");
    }
    
    // Validate customer_id
    if (formData.customer_id && (formData.customer_id.length > 100 || !/^[a-zA-Z0-9_-]*$/.test(formData.customer_id))) {
      errors.push("Customer ID must be alphanumeric with dashes/underscores and under 100 characters");
    }
    
    // Validate source
    if (!formData.source || formData.source.length > 50 || !/^[a-zA-Z0-9_-]+$/.test(formData.source)) {
      errors.push("Source is required and must be alphanumeric with dashes/underscores and under 50 characters");
    }
    
    // Validate raw_text
    if (!formData.raw_text || formData.raw_text.trim().length === 0) {
      errors.push("Raw text is required");
    } else if (formData.raw_text.length > 10000) {
      errors.push("Raw text must be under 10,000 characters");
    }
    
    // Validate sentiment
    if (formData.sentiment !== "") {
      const sentimentNum = Number(formData.sentiment);
      if (isNaN(sentimentNum) || sentimentNum < -1 || sentimentNum > 1) {
        errors.push("Sentiment must be a number between -1 and 1");
      }
    }
    
    return errors;
  };

  const sanitizeInput = (input) => {
    if (typeof input !== 'string') return input;
    return input.trim().replace(/[<>]/g, '');
  };

  const sanitizeApiResponse = (data) => {
    if (!data || typeof data !== 'object') return data;
    
    const sanitized = { ...data };
    
    // Sanitize string properties
    Object.keys(sanitized).forEach(key => {
      if (typeof sanitized[key] === 'string') {
        sanitized[key] = sanitized[key].replace(/<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>/gi, '');
      }
    });
    
    return sanitized;
  };

  const getGenericErrorMessage = (operation) => {
    const messages = {
      health: "Unable to check system health",
      conversations: "Unable to load conversations", 
      create: "Failed to create conversation",
      summary: "Failed to generate summary",
      escalate: "Escalation request failed"
    };
    return messages[operation] || "An error occurred";
  };

  // On load, ping /health
  useEffect(() => {
    fetch(`${API_BASE}/health`)
      .then((r) => r.json())
      .then((data) => setHealth(sanitizeApiResponse(data)))
      .catch(() => setError(getGenericErrorMessage("health")));
  }, []);

  async function loadRecent() {
    setError("");
    try {
      const r = await fetch(`${API_BASE}/conversations/recent`);
      const data = await r.json();
      setRecent((data.items || []).map(item => sanitizeApiResponse(item)));
    } catch (e) {
      setError(getGenericErrorMessage("conversations"));
    }
  }

  async function submitForm(e) {
    e.preventDefault();
    setError("");
    setPostResult(null);
    
    // Validate form data
    const validationErrors = validateForm(form);
    if (validationErrors.length > 0) {
      setError(validationErrors.join("; "));
      return;
    }
    
    try {
      const payload = {
        external_id: sanitizeInput(form.external_id) || undefined,
        customer_id: sanitizeInput(form.customer_id) || undefined,
        source: sanitizeInput(form.source),
        raw_text: sanitizeInput(form.raw_text),
        sentiment: form.sentiment === "" ? null : Number(form.sentiment),
      };
      
      const r = await fetch(`${API_BASE}/conversations`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      
      if (!r.ok) {
        throw new Error(`HTTP ${r.status}`);
      }
      
      const data = await r.json();
      setPostResult(sanitizeApiResponse(data));
      await loadRecent();
    } catch (e) {
      setError(getGenericErrorMessage("create"));
    }
  }

  // ---- Step 2.3: summaries ----
  async function generateSummary(rawText, k = 3) {
    const res = await fetch(`${API_BASE}/context/summary`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: rawText, k }),
    });
    if (!res.ok) throw new Error("summary_failed");
    const data = await res.json();
    return sanitizeApiResponse(data);
  }

  async function handleGenerateSummary() {
    if (!selectedConv?.raw_text) {
      setUiError("No conversation selected.");
      return;
    }
    setUiError("");
    setLoadingSummary(true);
    setSummary("");
    setSimilar([]);
    try {
      const data = await generateSummary(selectedConv.raw_text, 3);
      setSummary(data.summary || "");
      setSimilar(data.similar || []);
    } catch (e) {
      setUiError("Failed to generate summary");
    } finally {
      setLoadingSummary(false);
    }
  }

  // ---- Step 3.1: Escalate (call backend rule) ----
  async function escalate(id, force = false) {
    const res = await fetch(`${API_BASE}/escalate/check?id=${id}&force=${force}`, {
      method: "POST",
    });
    if (!res.ok) throw new Error("escalate_failed");
    const data = await res.json();
    return sanitizeApiResponse(data);
  }

  function update(field, value) {
    setForm((f) => ({ ...f, [field]: value }));
  }

  return (
    <div style={{ maxWidth: 800, margin: "30px auto", fontFamily: "system-ui, sans-serif" }}>
      <h1>Smart Escalation — Mini UI</h1>

      {/* Health */}
      <section style={{ padding: 16, border: "1px solid #ddd", borderRadius: 8, marginBottom: 20 }}>
        <h2>Health</h2>
        <pre>{health ? JSON.stringify(health, null, 2) : "Checking..."}</pre>
        {error && <p style={{ color: "crimson" }}>{error}</p>}
      </section>

      {/* Insert Conversation */}
      <section style={{ padding: 16, border: "1px solid #ddd", borderRadius: 8, marginBottom: 20 }}>
        <h2>Insert Conversation</h2>
        <form onSubmit={submitForm} style={{ display: "grid", gap: 8 }}>
          <input
            placeholder="external_id (optional)"
            value={form.external_id}
            onChange={(e) => update("external_id", e.target.value)}
          />
          <input
            placeholder="customer_id (optional)"
            value={form.customer_id}
            onChange={(e) => update("customer_id", e.target.value)}
          />
          <input
            placeholder="source"
            value={form.source}
            onChange={(e) => update("source", e.target.value)}
          />
          <textarea
            placeholder="raw_text"
            value={form.raw_text}
            onChange={(e) => update("raw_text", e.target.value)}
          />
          <input
            placeholder="sentiment (e.g., -0.6)"
            value={form.sentiment}
            onChange={(e) => update("sentiment", e.target.value)}
          />
          <button type="submit">Create</button>
        </form>
        {postResult && (
          <p style={{ marginTop: 10 }}>
            Created: <code>{JSON.stringify(postResult)}</code>
          </p>
        )}
      </section>

      {/* Recent + select + escalate */}
      <section style={{ padding: 16, border: "1px solid #ddd", borderRadius: 8 }}>
        <h2>Recent Conversations</h2>
        <button onClick={loadRecent}>Load recent</button>
        <ul style={{ marginTop: 10 }}>
          {recent.map((conv) => {
            const isSelected = selectedConv?.id === conv.id;
            return (
              <li key={conv.id} style={{ marginBottom: 12 }}>
                <strong>#{conv.id}</strong> — {conv.source} — sentiment: {String(conv.sentiment)}{" "}
                <button
                  style={{ marginLeft: 8 }}
                  onClick={() => {
                    setSelectedConv(conv);
                    setSummary("");
                    setSimilar([]);
                    setUiError("");
                  }}
                >
                  {isSelected ? "Selected" : "Select"}
                </button>
                <button
                  style={{ marginLeft: 8 }}
                  onClick={async () => {
                    try {
                      const r = await escalate(conv.id);
                      // Show the returned summary if any, and select this conversation so the panel appears
                      setSelectedConv(conv);
                      if (r.summary) {
                        setSummary(r.summary);
                        setSimilar(r.similar || []);
                      }
                      alert(r.escalate ? "Escalated" : `Not escalated: ${r.reason}`);
                    } catch (e) {
                      alert(getGenericErrorMessage("escalate"));
                    }
                  }}
                >
                  Escalate
                </button>
                <br />
                <em>{conv.raw_text}</em>
              </li>
            );
          })}
        </ul>
      </section>

      {/* Summary panel (appears when a conversation is selected) */}
      {selectedConv && (
        <section style={{ padding: 16, border: "1px solid #ddd", borderRadius: 8, marginTop: 20 }}>
          <h2>Generate Summary</h2>
          <p style={{ marginTop: 0 }}>
            Selected conversation: <strong>#{selectedConv.id}</strong>
          </p>
          <button onClick={handleGenerateSummary} disabled={loadingSummary}>
            {loadingSummary ? "Working..." : "Generate Summary"}
          </button>
          <button
            style={{ marginLeft: 8 }}
            onClick={() => {
              setSelectedConv(null);
              setSummary("");
              setSimilar([]);
              setUiError("");
            }}
          >
            Clear Selection
          </button>

          {uiError && <div style={{ color: "crimson", marginTop: 8 }}>{uiError}</div>}

          {summary && (
            <>
              <h3 style={{ marginTop: 16 }}>Summary</h3>
              <pre style={{ whiteSpace: "pre-wrap" }}>{summary}</pre>
              <button onClick={() => navigator.clipboard.writeText(summary)}>Copy</button>
            </>
          )}

          {similar.length > 0 && (
            <div style={{ marginTop: 16 }}>
              <h3>Top-3 Similar</h3>
              {similar.map((s) => (
                <div key={s.id} style={{ marginBottom: 10 }}>
                  <small>
                    score{" "}
                    {typeof s.score === "number" ? s.score.toFixed(3) : String(s.score)}
                  </small>
                  <div>{s.raw_text}</div>
                </div>
              ))}
            </div>
          )}
        </section>
      )}
    </div>
  );
}
