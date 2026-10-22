import { useCallback, useEffect, useRef, useState } from "react";
import {
  AlertCircle,
  ArrowDown,
  ArrowUp,
  CheckCircle2,
  ChevronDown,
  FileText,
  Info,
  LoaderCircle,
  MessageCircle,
  Plus,
  RefreshCw,
  Scale,
  Search,
  Sparkles,
  ThumbsDown,
  ThumbsUp,
} from "lucide-react";
import { askLegalQuery, getGatewayHealth } from "./api";
import { pipelineTelemetryRows, safeFailureTelemetry } from "./telemetry";
import "./App.css";

const HISTORY_STORAGE_KEY = "dai-legal-chat-history-v1";
const UNVERIFIED_FALLBACK = "Chua the dua ra cau tra loi phap ly du dieu kien kiem tra trich dan Evidence.";
const POLITE_UNVERIFIED_FALLBACK = "Dạ, xin lỗi bạn. Hiện hệ thống chưa đủ điều kiện để đưa ra câu trả lời pháp lý có trích dẫn Evidence đã được kiểm tra. Mong bạn thông cảm và thử lại sau ạ.";

function readConversationHistory() {
  try {
    const saved = JSON.parse(localStorage.getItem(HISTORY_STORAGE_KEY) || "[]");
    return Array.isArray(saved)
      ? saved.filter((item) => item && typeof item.id === "string" && Array.isArray(item.messages))
      : [];
  } catch {
    return [];
  }
}

const QUICK_QUESTIONS = [
  "Thủ tục thành lập công ty cổ phần là gì?",
  "Đăng ký tạm trú cần những giấy tờ gì?",
  "Đăng ký khai sinh cần thực hiện thế nào?",
  "Cấp phiếu lý lịch tư pháp cần làm gì?",
];

const VERIFIER_LABELS = {
  passed: { label: "Đã kiểm tra liên kết citation", tone: "success" },
  failed: { label: "Không đạt kiểm tra citation", tone: "warning" },
  error: { label: "Không thể kiểm tra citation", tone: "warning" },
  not_required: { label: "Không yêu cầu kiểm tra", tone: "neutral" },
  not_reported: { label: "API không gửi trạng thái kiểm tra", tone: "neutral" },
};

function makeLocalId(prefix) {
  return `${prefix}-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

function formatTime(value) {
  if (!value) return "";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "";
  return new Intl.DateTimeFormat("vi-VN", {
    hour: "2-digit",
    minute: "2-digit",
  }).format(date);
}

function CitationText({ content, pipeline }) {
  const evidenceIds = Array.isArray(pipeline?.evidence_ids)
    ? pipeline.evidence_ids
    : null;
  const segments = content.split(/(\[Evidence ID:\s*[^\]]+\])/gi);

  return segments.map((segment, index) => {
    const match = segment.match(/^\[Evidence ID:\s*([^\]]+)\]$/i);
    if (!match) return <span key={index}>{segment}</span>;

    const evidenceId = match[1].trim();
    const known = evidenceIds?.includes(evidenceId);
    const className = known === false ? "citation-chip citation-chip--unknown" : "citation-chip";

    return (
      <span
        className={className}
        key={`${evidenceId}-${index}`}
        title={known === false ? "Evidence ID không có trong danh sách API cung cấp" : "Evidence ID do câu trả lời cung cấp"}
      >
        <FileText size={13} aria-hidden="true" />
        {evidenceId}
      </span>
    );
  });
}

function PipelineDetails({ message }) {
  const pipeline = message.pipeline || {};
  const telemetryRows = pipelineTelemetryRows(pipeline, message.requestId);
  if (Object.keys(pipeline).length === 0 && telemetryRows.length === 0) return null;

  const evidenceIds = Array.isArray(pipeline.evidence_ids) ? pipeline.evidence_ids : [];
  const citedIds = Array.isArray(pipeline.cited_evidence_ids)
    ? pipeline.cited_evidence_ids
    : [];
  const rows = [
    ["Truy xuất", pipeline.retrieval],
    ["Xếp hạng lại", pipeline.reranking],
    ["Evidence Builder", pipeline.evidence_builder],
    ["Độ đầy đủ dữ liệu nội bộ", pipeline.enough_data],
    ["Nguồn ngoài", pipeline.external_fallback],
    ["Model API", pipeline.model_api],
    ["Evidence trong prompt", pipeline.evidence_count],
    ["Kiểm tra citation", pipeline.citation_verification || message.verifier],
    ["Số lần tạo", pipeline.citation_attempts],
    ["Trạng thái Gateway", message.responseStatus],
    ["Thời gian Model API", pipeline.model_processing_time_ms == null ? undefined : `${pipeline.model_processing_time_ms} ms`],
    ["Thời gian gọi Model API", pipeline.model_api_elapsed_ms == null ? undefined : `${pipeline.model_api_elapsed_ms} ms`],
    ["Domain", message.routing?.selected_domain],
  ].filter(([, value]) => value !== undefined && value !== null);

  return (
    <details className="pipeline-details">
      <summary>
        <span><Info size={15} /> Chi tiết xử lý từ API</span>
        <ChevronDown size={15} className="details-chevron" />
      </summary>
      <div className="pipeline-content">
        <dl className="pipeline-grid">
          {telemetryRows.map(([label, value]) => (
            <div className="pipeline-row" key={`telemetry-${label}`}>
              <dt>{label}</dt>
              <dd>{value}</dd>
            </div>
          ))}
          {rows.map(([label, value]) => (
            <div className="pipeline-row" key={label}>
              <dt>{label}</dt>
              <dd>{String(value)}</dd>
            </div>
          ))}
          {message.verifierScope && message.verifierScope !== "not_reported" && (
            <div className="pipeline-row">
              <dt>Phạm vi verifier</dt>
              <dd>{message.verifierScope}</dd>
            </div>
          )}
        </dl>

        {evidenceIds.length > 0 && (
          <div className="evidence-group">
            <span className="evidence-label">Evidence ID trong prompt</span>
            <div className="evidence-list">
              {evidenceIds.map((id) => (
                <code key={id}>{id}</code>
              ))}
            </div>
          </div>
        )}
        {citedIds.length > 0 && (
          <div className="evidence-group">
            <span className="evidence-label">Evidence ID parser nhận được</span>
            <div className="evidence-list">
              {citedIds.map((id) => (
                <code key={id}>{id}</code>
              ))}
            </div>
          </div>
        )}
        {(pipeline.required_fields?.length > 0 || pipeline.missing_fields?.length > 0) && (
          <div className="evidence-group">
            <span className="evidence-label">Trường dữ liệu được kiểm tra</span>
            <p>
              Cần: {(pipeline.required_fields || []).join(", ") || "không có"}
              {pipeline.missing_fields?.length > 0
                ? ` · Còn thiếu: ${pipeline.missing_fields.join(", ")}`
                : " · Không thiếu trường được nhận diện"}
            </p>
          </div>
        )}
        {Array.isArray(message.executionPlan) && message.executionPlan.length > 0 && (
          <div className="evidence-group">
            <span className="evidence-label">Execution plan từ Gateway</span>
            <div className="evidence-list">
              {message.executionPlan.map((step, index) => (
                <code key={`${step.step ?? index}-${step.action ?? "step"}`}>
                  {step.step ?? index + 1}. {step.action ?? ""} — {step.status ?? ""}
                </code>
              ))}
            </div>
          </div>
        )}
        <p className="pipeline-note">
          API hiện chỉ gửi mã Evidence, không gửi nội dung hoặc liên kết nguồn để hiển thị tại đây.
          Trạng thái citation không xác nhận câu trả lời đúng về mặt ngữ nghĩa.
        </p>
      </div>
    </details>
  );
}

function AssistantMessage({ message, onFeedback }) {
  const verifier = VERIFIER_LABELS[message.verifier || "not_reported"] || VERIFIER_LABELS.not_reported;
  const hasAnswer = Boolean(message.content?.trim());
  const displayedAnswer = message.content?.trim() === UNVERIFIED_FALLBACK
    ? POLITE_UNVERIFIED_FALLBACK
    : message.content;

  return (
    <article className="message-row message-row--assistant">
      <div className="assistant-avatar"><Scale size={17} /></div>
      <div className="message-column">
        <div className="message-heading">
          <strong>Trợ lý pháp lý</strong>
          {message.created_at && <time>{formatTime(message.created_at)}</time>}
          {message.local && <span className="live-tag">Vừa xong</span>}
        </div>
        <div className={`assistant-card${message.error ? " assistant-card--error" : ""}`}>
          {message.loading ? (
            <div className="thinking-state" role="status">
              <LoaderCircle className="spin" size={17} />
              <span>Đang tra cứu và tạo câu trả lời…</span>
            </div>
          ) : message.error ? (
            <div className="inline-error" role="alert">
              <AlertCircle size={17} />
              <div>
                <strong>Chưa thể kết nối tới dịch vụ</strong>
                <p>{message.error}</p>
                {message.retryQuestion && (
                  <button className="text-button" onClick={() => message.onRetry?.()} type="button">
                    <RefreshCw size={14} /> Thử lại câu hỏi
                  </button>
                )}
              </div>
              {safeFailureTelemetry(message.failureTelemetry).length > 0 && (
                <details className="pipeline-details">
                  <summary><span><Info size={15} /> Chi tiết xử lý từ API</span></summary>
                  <div className="pipeline-content"><dl className="pipeline-grid">
                    {safeFailureTelemetry(message.failureTelemetry).map(([label, value]) => (
                      <div className="pipeline-row" key={label}><dt>{label}</dt><dd>{value}</dd></div>
                    ))}
                  </dl></div>
                </details>
              )}
            </div>
          ) : !hasAnswer ? (
            <div className="empty-answer" role="status">
              API đã phản hồi nhưng không có nội dung câu trả lời.
            </div>
          ) : (
            <>
              {(message.responseStatus === "FAILED" || (message.verifier && message.verifier !== "passed")) && (
                <div className={`verification-banner verification-banner--${message.responseStatus === "FAILED" ? "warning" : verifier.tone}`} role="status">
                  {message.responseStatus === "FAILED" || verifier.tone === "warning" ? <AlertCircle size={16} /> : <Info size={16} />}
                  <span>{message.responseStatus === "FAILED" ? "Gateway báo request FAILED; câu trả lời có thể là fallback an toàn." : verifier.label}</span>
                </div>
              )}
              <div className="answer-content">
                <CitationText content={displayedAnswer} pipeline={message.pipeline} />
              </div>
              {message.verifier === "passed" && (
                <div className="verification-caption">
                  <CheckCircle2 size={14} /> Đã kiểm tra liên kết Evidence ID; chưa đánh giá hỗ trợ ngữ nghĩa.
                </div>
              )}
              {message.requestId && <div className="verification-caption">Request {message.requestId}{Number.isFinite(message.processingTimeMs) ? ` · Gateway ${message.processingTimeMs} ms` : ""}</div>}
              <PipelineDetails message={message} />
              <div className="message-feedback" aria-label="Đánh giá câu trả lời">
                <span>Câu trả lời này hữu ích?</span>
                <button
                  className={`feedback-button${message.feedback === "like" ? " feedback-button--selected" : ""}`}
                  type="button"
                  aria-label="Câu trả lời hữu ích"
                  aria-pressed={message.feedback === "like"}
                  onClick={() => onFeedback(message.id, "like")}
                >
                  <ThumbsUp size={15} /> Hữu ích
                </button>
                <button
                  className={`feedback-button${message.feedback === "dislike" ? " feedback-button--selected feedback-button--dislike" : ""}`}
                  type="button"
                  aria-label="Câu trả lời chưa hữu ích"
                  aria-pressed={message.feedback === "dislike"}
                  onClick={() => onFeedback(message.id, "dislike")}
                >
                  <ThumbsDown size={15} /> Chưa hữu ích
                </button>
              </div>
            </>
          )}
        </div>
      </div>
    </article>
  );
}

function UserMessage({ message }) {
  return (
    <article className="message-row message-row--user">
      <div className="user-card">
        <div className="message-heading"><strong>Bạn</strong>{message.created_at && <time>{formatTime(message.created_at)}</time>}</div>
        <p>{message.content}</p>
      </div>
    </article>
  );
}

function App() {
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState([]);
  const [historyEntries, setHistoryEntries] = useState(readConversationHistory);
  const [activeConversationId, setActiveConversationId] = useState(() => makeLocalId("conversation"));
  const [loading, setLoading] = useState(false);
  const [healthLoading, setHealthLoading] = useState(true);
  const [connectionError, setConnectionError] = useState("");
  const [allowExternal, setAllowExternal] = useState(false);
  const [showScrollButton, setShowScrollButton] = useState(false);
  const scrollRef = useRef(null);
  const inputRef = useRef(null);

  const checkGateway = useCallback(async () => {
    setHealthLoading(true);
    setConnectionError("");
    try {
      await getGatewayHealth();
    } catch (error) {
      setConnectionError(error.message || "Không thể kết nối Core Gateway.");
    } finally {
      setHealthLoading(false);
    }
  }, []);

  useEffect(() => { checkGateway(); }, [checkGateway]);

  useEffect(() => {
    try {
      localStorage.setItem(HISTORY_STORAGE_KEY, JSON.stringify(historyEntries));
    } catch {
      // Keep the current conversation usable when browser storage is unavailable.
    }
  }, [historyEntries]);

  useEffect(() => {
    if (!messages.length || messages.some((message) => message.loading)) return;
    const firstQuestion = messages.find((message) => message.role === "user")?.content;
    if (!firstQuestion) return;

    const entry = {
      id: activeConversationId,
      title: firstQuestion,
      updatedAt: new Date().toISOString(),
      messages,
    };
    setHistoryEntries((current) => [
      entry,
      ...current.filter((item) => item.id !== activeConversationId),
    ].slice(0, 30));
  }, [messages, activeConversationId]);

  useEffect(() => {
    const element = scrollRef.current;
    if (!element) return;
    const nearBottom = element.scrollHeight - element.scrollTop - element.clientHeight < 140;
    if (nearBottom) element.scrollTo({ top: element.scrollHeight, behavior: "smooth" });
  }, [messages, loading]);

  const updateMessage = (id, update) => {
    setMessages((current) => current.map((message) =>
      message.id === id ? { ...message, ...update } : message,
    ));
  };

  const sendQuestion = async (value, { retry = false } = {}) => {
    const prompt = value.trim();
    if (!prompt || loading) return;

    setQuestion("");
    setLoading(true);
    setConnectionError("");

    const userId = makeLocalId("user");
    const assistantId = makeLocalId("assistant");

    setMessages((current) => [
      ...current,
      ...(!retry ? [{ id: userId, role: "user", content: prompt, local: true }] : []),
      { id: assistantId, role: "assistant", content: "", loading: true, local: true },
    ]);

    try {
      const result = await askLegalQuery(prompt, { allowExternal });

      updateMessage(assistantId, {
        content: result.answer,
        loading: false,
        verifier: result.pipeline_execution?.verifier || "not_reported",
        verifierScope: result.pipeline_execution?.verifier_scope || "not_reported",
        pipeline: result.pipeline_execution || {},
        routing: result.routing_and_planning || {},
        executionPlan: result.routing_and_planning?.execution_plan,
        responseStatus: result.status,
        requestId: result.request_id,
        processingTimeMs: result.processing_time_ms,
        local: true,
      });
    } catch (error) {
      updateMessage(assistantId, {
        loading: false,
        error: error.message || "Đã xảy ra lỗi khi gửi câu hỏi.",
        failureTelemetry: error.telemetry,
        retryQuestion: prompt,
        onRetry: () => sendQuestion(prompt, { retry: true }),
      });
    } finally {
      setLoading(false);
      inputRef.current?.focus();
    }
  };

  const handleSubmit = (event) => {
    event.preventDefault();
    sendQuestion(question);
  };

  const startNewConversation = () => {
    setActiveConversationId(makeLocalId("conversation"));
    setMessages([]);
    setQuestion("");
    inputRef.current?.focus();
  };

  const openConversation = (entry) => {
    setActiveConversationId(entry.id);
    setMessages(entry.messages.map((message) => ({
      ...message,
      loading: false,
      onRetry: undefined,
      retryQuestion: undefined,
    })));
    setQuestion("");
  };

  const handleScroll = () => {
    const element = scrollRef.current;
    if (!element) return;
    setShowScrollButton(element.scrollHeight - element.scrollTop - element.clientHeight > 220);
  };

  const isEmpty = messages.length === 0;
  return (
    <div className="app-shell">
      <aside className="sidebar">
        <a className="brand" href="#main" aria-label="AI Pháp Lý - về đầu trang">
          <span className="brand-mark"><Scale size={22} /></span>
          <span><strong>AI Pháp Lý</strong><small>DAI-Legal-Model</small></span>
        </a>

          <button className="new-question-button" onClick={startNewConversation} disabled={loading} type="button">
          <Plus size={17} /> Câu hỏi mới
        </button>

        <div className="sidebar-section">
          <span className="eyebrow">LỊCH SỬ CÂU HỎI</span>
          <div className="history-list">
            {historyEntries.length === 0 ? (
              <p className="history-empty">Câu hỏi gần đây sẽ hiển thị ở đây.</p>
            ) : historyEntries.map((entry) => (
              <button
                className={`history-item${entry.id === activeConversationId ? " history-item--active" : ""}`}
                key={entry.id}
                onClick={() => openConversation(entry)}
                disabled={loading}
                title={entry.title}
                type="button"
              >
                <MessageCircle size={14} />
                <span>{entry.title}</span>
              </button>
            ))}
          </div>
        </div>

        <div className="sidebar-spacer" />
      </aside>

      <main className="workspace" id="main">
        <section className="chat-layout" aria-label="Trợ lý pháp lý">
          <div className="chat-title-row">
            <div>
              <div className="section-kicker"><Sparkles size={15} /> HỖ TRỢ TRA CỨU PHÁP LUẬT</div>
              <h1>Xin chào, bạn cần tìm hiểu điều gì?</h1>
            </div>
            <div className="chat-actions">
              <button className="icon-button" onClick={checkGateway} disabled={healthLoading || loading} title="Kiểm tra Gateway" type="button">
                <RefreshCw size={17} className={healthLoading ? "spin" : ""} />
                <span className="sr-only">Kiểm tra Gateway</span>
              </button>
              <button className="icon-button" onClick={startNewConversation} disabled={loading || messages.length === 0} title="Bắt đầu phiên mới" type="button">
                <Plus size={17} />
                <span className="sr-only">Bắt đầu phiên mới</span>
              </button>
            </div>
          </div>

          {connectionError && (
            <div className="connection-error" role="alert">
              <AlertCircle size={18} />
              <div><strong>Không thể kết nối Core Gateway</strong><p>{connectionError}</p></div>
              <button onClick={checkGateway} type="button">Thử kết nối lại</button>
            </div>
          )}

          <div className="chat-card">
            <div className="chat-card-header">
              <div className="chat-card-icon"><MessageCircle size={18} /></div>
              <div><strong>Cuộc trò chuyện</strong><span>Tin nhắn chỉ lưu trong phiên giao diện hiện tại</span></div>
              <div className="session-label"><span /> Phiên hiện tại</div>
            </div>

            <div className="conversation" ref={scrollRef} onScroll={handleScroll} aria-live="polite">
              {isEmpty ? (
                <div className="welcome-state">
                  <div className="welcome-icon"><Scale size={29} /></div>
                  <span className="welcome-overline">TRỢ LÝ PHÁP LÝ AI</span>
                  <h2>Bắt đầu với một câu hỏi</h2>
                  <p>Hãy nêu thủ tục hoặc vấn đề bạn muốn tìm hiểu. Câu trả lời sẽ hiển thị trạng thái citation nếu API cung cấp.</p>
                  <div className="suggestion-grid">
                    {QUICK_QUESTIONS.map((item) => (
                      <button key={item} className="suggestion-card" onClick={() => { setQuestion(item); inputRef.current?.focus(); }} type="button">
                        <Search size={16} /><span>{item}</span><ArrowUp size={15} className="suggestion-arrow" />
                      </button>
                    ))}
                  </div>
                </div>
              ) : (
                <div className="message-list">
                  {messages.map((message) => message.role === "user"
                    ? <UserMessage key={message.id} message={message} />
                    : <AssistantMessage
                        key={message.id}
                        message={message}
                        onFeedback={(id, feedback) => setMessages((current) => current.map((item) => (
                          item.id === id
                            ? { ...item, feedback: item.feedback === feedback ? null : feedback }
                            : item
                        )))}
                      />,
                  )}
                </div>
              )}
              {showScrollButton && (
                <button className="scroll-bottom" onClick={() => scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" })} type="button" aria-label="Cuộn xuống cuối">
                  <ArrowDown size={17} />
                </button>
              )}
            </div>

            {messages.length > 0 && (
              <div className="quick-question-row" aria-label="Câu hỏi gợi ý">
                {QUICK_QUESTIONS.slice(0, 3).map((item) => (
                  <button key={item} onClick={() => { setQuestion(item); inputRef.current?.focus(); }} type="button">
                    <Sparkles size={12} /> {item}
                  </button>
                ))}
              </div>
            )}

            <div className="composer-wrap">
              <form className="composer" onSubmit={handleSubmit}>
                <textarea
                  ref={inputRef}
                  value={question}
                  onChange={(event) => setQuestion(event.target.value)}
                  onKeyDown={(event) => {
                    if (event.key === "Enter" && !event.shiftKey) {
                      event.preventDefault();
                      handleSubmit(event);
                    }
                  }}
                  placeholder="Nhập câu hỏi pháp lý của bạn…"
                  rows={2}
                  disabled={loading}
                  aria-label="Câu hỏi pháp lý"
                />
                <div className="composer-controls">
                  <label className="external-toggle" title="Cho phép Core API tra cứu nguồn ngoài nếu pipeline yêu cầu">
                    <input type="checkbox" checked={allowExternal} onChange={(event) => setAllowExternal(event.target.checked)} disabled={loading} />
                    <span className="toggle-track"><span /></span>
                    <span>Cho phép nguồn ngoài</span>
                  </label>
                  <span className="keyboard-hint">Enter gửi · Shift + Enter xuống dòng</span>
                  <button className="send-button" type="submit" disabled={loading || !question.trim()} aria-label="Gửi câu hỏi">
                    {loading ? <LoaderCircle className="spin" size={17} /> : <ArrowUp size={18} />}
                    <span>{loading ? "Đang xử lý" : "Gửi câu hỏi"}</span>
                  </button>
                </div>
              </form>
            </div>
          </div>

          <footer className="page-footer">© 2026 DAI-Legal-Model <span>·</span> Công cụ hỗ trợ tra cứu, không thay thế tư vấn pháp lý chuyên môn.</footer>
        </section>
      </main>
    </div>
  );
}

export default App;
