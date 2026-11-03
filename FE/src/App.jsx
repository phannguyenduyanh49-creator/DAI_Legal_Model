import { useCallback, useEffect, useRef, useState } from "react";
import {
  AlertCircle,
  ArrowDown,
  ArrowUp,
  CheckCircle2,
  ChevronDown,
  ChevronUp,
  FileText,
  Info,
  LoaderCircle,
  MessageCircle,
  Plus,
  RefreshCw,
  Scale,
  Sparkles,
  ThumbsDown,
  ThumbsUp,
  Upload,
} from "lucide-react";
import { askLegalQuery, getDomains, getDomainReadiness, getGatewayHealth, getUploadDomains, uploadDomainDocument } from "./api";
import { answerOutcomeNotice, citationMarkersAreTrusted, stripUnverifiedCitationMarkers } from "./answerPresentation";
import { parseAnswerBlocks, parseInlineMarkdown } from "./answerMarkdown";
import { resolveCitationTitle, visibleEvidenceSources } from "./sourcePresentation";
import { getAvailableUploadDomains, loadUploadDomainOptions } from "./domainOptions";
import { domainOptionLabel, getDomainChatOptions } from "./domainChatOptions";
import {
  pipelineTelemetryRows,
  safeCitationAttemptDiagnostics,
  safeFailureTelemetry,
} from "./telemetry";
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

function groundingLabel(status) {
  return ({
    passed: "các claim được đánh giá là có Evidence hỗ trợ",
    failed: "có claim chưa được Evidence hỗ trợ đầy đủ",
    insufficient_evidence: "Evidence chưa đủ để đánh giá claim",
    contradicted: "có claim mâu thuẫn với Evidence",
    unsupported: "có claim chưa được Evidence hỗ trợ",
    partially_supported: "có claim chỉ được Evidence hỗ trợ một phần",
    inconclusive: "scorer chưa thể kết luận",
    evaluator_unavailable: "scorer không khả dụng",
    not_evaluated: "chưa được đánh giá",
  })[status] || "chưa được API báo cáo";
}

function readinessDescription(readiness) {
  if (!readiness) return "Chưa có trạng thái readiness; chưa xác minh runtime.";
  if (!readiness.enabled) return "Domain đã bị tắt trong Registry.";
  if (
    readiness.configured
    && readiness.data_available === "available"
    && readiness.retrieval_ready === "available"
    && readiness.model_ready === "available"
    && readiness.e2e_verified === "verified"
  ) return "Đã xác minh E2E.";
  if (!readiness.configured) return "Cấu hình domain chưa đầy đủ.";
  return readiness.reason || "Cấu hình đã đăng ký; dữ liệu, retrieval và model chưa được xác minh.";
}

function domainRiskNotice(domainId, riskLevel) {
  if (!domainId || !["high", "critical"].includes(riskLevel)) return null;
  if (domainId === "healthcare" && riskLevel === "critical") {
    return "Câu hỏi có dấu hiệu tình huống y tế khẩn cấp. Nếu đây là triệu chứng đang xảy ra, hãy liên hệ dịch vụ cấp cứu tại địa phương ngay; hệ thống không thể chẩn đoán.";
  }
  if (domainId === "healthcare") {
    return "Thông tin này không thay thế chẩn đoán hoặc hướng dẫn điều trị của nhân viên y tế.";
  }
  if (domainId === "food") {
    return "Hãy xác minh dị ứng và an toàn thực phẩm với nguồn/chuyên gia phù hợp; nội dung này không tự xác nhận an toàn.";
  }
  return "Thông tin có thể phụ thuộc thời điểm hoặc hoàn cảnh; hãy kiểm tra ngày và điều kiện trong nguồn được trích dẫn.";
}

function uploadAcceptValue(extensions) {
  const mimeTypes = {
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
  };
  return extensions.flatMap((extension) => [extension, mimeTypes[extension]])
    .filter(Boolean)
    .join(",");
}

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
  if (!citationMarkersAreTrusted(pipeline)) {
    return <>{stripUnverifiedCitationMarkers(content)}</>;
  }
  const evidenceSources = Array.isArray(pipeline?.evidence_sources) ? pipeline.evidence_sources : [];
  const segments = content.split(/(\[Evidence ID:\s*[^\]]+\])/gi);

  return segments.map((segment, index) => {
    const match = segment.match(/^\[Evidence ID:\s*([^\]]+)\]$/i);
    if (!match) return <span key={index}>{segment}</span>;
    const title = resolveCitationTitle(match[1].trim(), evidenceSources);
    if (!title) return null;
    return <span className="citation-chip" key={index}><FileText size={13} aria-hidden="true" />{title}</span>;
  });
}

function FormattedAnswer({ content, pipeline }) {
  const renderInline = (text) => parseInlineMarkdown(text).map((token, tokenIndex) => (
    token.type === "strong"
      ? <strong key={`strong-${tokenIndex}`}><CitationText content={token.text} pipeline={pipeline} /></strong>
      : <CitationText key={`text-${tokenIndex}`} content={token.text} pipeline={pipeline} />
  ));
  return (
    <div className="answer-content">
      {parseAnswerBlocks(content).map((block, index) => {
        if (block.type === "heading") {
          const Heading = block.level === 1 ? "h2" : block.level === 2 ? "h3" : "h4";
          return <Heading key={`heading-${index}`}>{renderInline(block.text)}</Heading>;
        }
        if (block.type === "unordered-list" || block.type === "ordered-list") {
          const List = block.type === "unordered-list" ? "ul" : "ol";
          return (
            <List key={`list-${index}`}>
              {block.items.map((item, itemIndex) => (
                <li key={`item-${itemIndex}`}>{renderInline(item)}</li>
              ))}
            </List>
          );
        }
        return <p key={`paragraph-${index}`}>{renderInline(block.text)}</p>;
      })}
    </div>
  );
}

function PipelineDetails({ message }) {
  const pipeline = message.pipeline || {};
  const telemetryRows = pipelineTelemetryRows(pipeline, message.requestId);
  if (Object.keys(pipeline).length === 0 && telemetryRows.length === 0) return null;

  const evidenceIds = Array.isArray(pipeline.evidence_ids) ? pipeline.evidence_ids : [];
  const citedIds = Array.isArray(pipeline.cited_evidence_ids) ? pipeline.cited_evidence_ids : [];
  const evidenceSources = visibleEvidenceSources(pipeline.evidence_sources);
  const citationAttempts = safeCitationAttemptDiagnostics(
    pipeline.citation_attempt_diagnostics,
  );
  const rows = [
    ["Truy xuất", pipeline.retrieval],
    ["Collection truy xuất", pipeline.retrieval_collection_status],
    ["Xếp hạng lại", pipeline.reranking],
    ["Evidence Builder", pipeline.evidence_builder],
    ["Answer evidence sufficiency", pipeline.answer_data_sufficiency],
    ["Độ đầy đủ dữ liệu nội bộ", pipeline.enough_data],
    ["Nguồn ngoài", pipeline.external_fallback],
    ["Model API", pipeline.model_api],
    ["Evidence trong prompt", pipeline.evidence_count],
    ["Tính toàn vẹn Evidence", pipeline.evidence_integrity_status],
    ["Kiểm tra citation", pipeline.citation_verification || message.verifier],
    ["Grounding ngữ nghĩa", pipeline.claim_grounding_status],
    ["Trạng thái evaluator grounding", pipeline.grounding_scorer_status],
    ["Confidence", pipeline.confidence_status],
    ["Điểm confidence", pipeline.confidence_score],
    ["Kết quả verifier", pipeline.verification_outcome],
    ["Số lần tạo", pipeline.citation_attempts],
    ["Trạng thái Gateway", message.responseStatus],
    ["Thời gian Gateway", message.processingTimeMs == null ? undefined : `${message.processingTimeMs} ms`],
    ["Thời gian Model API", pipeline.model_processing_time_ms == null ? undefined : `${pipeline.model_processing_time_ms} ms`],
    ["Thời gian gọi Model API", pipeline.model_api_elapsed_ms == null ? undefined : `${pipeline.model_api_elapsed_ms} ms`],
    ["Domain", message.routing?.selected_domain],
    ["Mức rủi ro domain", message.domainPolicy?.risk_level],
    ["Lý do phân loại rủi ro", message.domainPolicy?.risk_reason_code],
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
            <span className="evidence-label">Số nguồn Evidence trong prompt: {evidenceIds.length}</span>
          </div>
        )}
        {evidenceSources.length > 0 && (
          <div className="evidence-group">
            <span className="evidence-label">Nguồn Evidence</span>
            <div className="evidence-list">
              {evidenceSources.map((source, index) => (
                  <div className="evidence-source" key={index}>
                    {source.title && <strong>{source.title}</strong>}
                    {source.location && <span>{source.location}</span>}
                    {source.sourceUrl && <a href={source.sourceUrl} target="_blank" rel="noreferrer">Mở nguồn</a>}
                  </div>
              ))}
            </div>
          </div>
        )}
        {citedIds.length > 0 && (
          <div className="evidence-group">
            <span className="evidence-label">Số nguồn được trích dẫn: {citedIds.length}</span>
          </div>
        )}
        {citationAttempts.length > 0 && (
          <div className="evidence-group">
            <span className="evidence-label">Chẩn đoán citation theo lần thử</span>
            <div className="evidence-list">
              {citationAttempts.map((attempt) => (
                <code key={attempt.attempt}>
                  Lần {attempt.attempt}: {attempt.claim_count ?? "?"} claims, {attempt.claims_with_citation ?? "?"} có citation, {attempt.cited_evidence_count ?? "?"}/{attempt.allowed_evidence_count ?? "?"} Evidence ID, {attempt.unknown_evidence_count ?? "?"} ID lạ
                  {attempt.generation_elapsed_ms != null ? ` · ${attempt.generation_elapsed_ms} ms tổng` : ""}
                  {attempt.model_processing_time_ms != null ? ` · ${attempt.model_processing_time_ms} ms model` : ""}
                  {attempt.parser_errors.length ? ` · parser: ${attempt.parser_errors.join(", ")}` : " · parser: không lỗi cú pháp"}
                  {attempt.verifier_reasons.length ? ` · verifier: ${attempt.verifier_reasons.join(", ")}` : ""}
                  {attempt.claim_grounding_status ? ` · grounding: ${attempt.claim_grounding_status}` : ""}
                  {attempt.confidence_status ? ` · confidence: ${attempt.confidence_status}` : ""}
                </code>
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
          Chỉ metadata nguồn do API cung cấp mới được hiển thị; liên kết nguồn có thể không có sẵn.
          Trạng thái citation không xác nhận câu trả lời đúng về mặt ngữ nghĩa.
        </p>
      </div>
    </details>
  );
}

function AssistantMessage({ message, onFeedback }) {
  const outcomeNotice = answerOutcomeNotice(message.responseStatus, message.verifier, message.pipeline);
  const hasAnswer = Boolean(message.content?.trim());
  const displayedAnswer = message.content?.trim() === UNVERIFIED_FALLBACK
    ? POLITE_UNVERIFIED_FALLBACK
    : message.content;
  const riskNotice = domainRiskNotice(
    message.domainPolicy?.domain_id || message.routing?.selected_domain,
    message.domainPolicy?.risk_level || message.routing?.risk_level,
  );

  return (
    <article className="message-row message-row--assistant">
      <div className="assistant-avatar"><Scale size={17} /></div>
      <div className="message-column">
        <div className="message-heading">
          <strong>Trợ lý pháp lý</strong>
          {message.created_at && <time>{formatTime(message.created_at)}</time>}
          {message.local && <span className="live-tag">Vừa xong</span>}
        </div>
        {(message.domainPolicy?.domain_id || message.routing?.selected_domain) && (
          <div className="message-domain" aria-label="Domain được sử dụng">
            Domain được sử dụng: <strong>{message.domainPolicy?.domain_id || message.routing.selected_domain}</strong>
          </div>
        )}
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
              {riskNotice && (
                <div className="verification-banner verification-banner--warning" role="status">
                  <AlertCircle size={16} />
                  <span>{riskNotice}</span>
                </div>
              )}
              {outcomeNotice && (
                <div className={`verification-banner verification-banner--${outcomeNotice.tone}`} role="status">
                  <AlertCircle size={16} />
                  <span>{outcomeNotice.text}</span>
                </div>
              )}
              <FormattedAnswer content={displayedAnswer} pipeline={message.pipeline} />
              {(message.pipeline?.failure_stage === "model_api"
                || message.pipeline?.model_api === "error"
                || message.pipeline?.failure_reason_code === "MODEL_API_REQUEST_FAILED")
                && message.retryQuestion && (
                  <button className="text-button" onClick={() => message.onRetry?.()} type="button">
                    <RefreshCw size={14} /> Thử lại câu hỏi
                  </button>
                )}
              {message.pipeline?.citation_verification === "passed" && (
                <div className="verification-caption">
                  {message.pipeline?.verification_status === "passed"
                    && message.pipeline?.claim_grounding_status === "passed"
                    && message.pipeline?.verifier_scope?.includes("semantic_grounding")
                    ? <CheckCircle2 size={14} />
                    : <Info size={14} />}
                  <span>Liên kết citation hợp lệ; {groundingLabel(message.pipeline?.claim_grounding_status)}.</span>
                </div>
              )}
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
  const [domains, setDomains] = useState([]);
  const [uploadDomains, setUploadDomains] = useState([]);
  const [uploadRegistryError, setUploadRegistryError] = useState(false);
  const [uploadRegistryErrorMessage, setUploadRegistryErrorMessage] = useState("");
  const [uploadRegistryDiagnostic, setUploadRegistryDiagnostic] = useState(null);
  const [uploadRegistryLoading, setUploadRegistryLoading] = useState(true);
  const [uploadRegistrySource, setUploadRegistrySource] = useState("loading");
  const [domainReadiness, setDomainReadiness] = useState({});
  const [domainHint, setDomainHint] = useState("");
  const [responseStyle, setResponseStyle] = useState("default");
  const [uploadDomainId, setUploadDomainId] = useState("");
  const [domainToolsOpen, setDomainToolsOpen] = useState(false);
  const [uploadFile, setUploadFile] = useState(null);
  const [uploadStatus, setUploadStatus] = useState("");
  const [uploadStatusKind, setUploadStatusKind] = useState("ready");
  const [uploading, setUploading] = useState(false);
  const [fileDropActive, setFileDropActive] = useState(false);
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
    getDomains().then(setDomains).catch(() => setDomains([]));
    loadUploadDomains();
    getDomainReadiness().then((result) => {
      setDomainReadiness(Object.fromEntries(result.domains.map((domain) => [domain.domain_id, domain])));
    }).catch(() => {
      setDomainReadiness({});
    });
  }, []);

  useEffect(() => {
    const textarea = inputRef.current;
    if (!textarea) return;
    const maxHeight = 150;
    textarea.style.height = "auto";
    textarea.style.height = `${Math.min(Math.max(textarea.scrollHeight, 48), maxHeight)}px`;
    textarea.style.overflowY = textarea.scrollHeight > maxHeight ? "auto" : "hidden";
  }, [question]);

  const loadUploadDomains = async () => {
    setUploadRegistryLoading(true);
    setUploadRegistryError(false);
    setUploadRegistryErrorMessage("");
    const result = await loadUploadDomainOptions(getUploadDomains);
    setUploadDomains(result.domains);
    setUploadRegistrySource(result.source);
    setUploadRegistryError(Boolean(result.error));
    setUploadRegistryErrorMessage(result.error);
    setUploadRegistryDiagnostic(result.diagnostic);
    setUploadRegistryLoading(false);
  };

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
      const result = await askLegalQuery(prompt, { allowExternal, domainHint, responseStyle });

      updateMessage(assistantId, {
        content: result.answer,
        loading: false,
        verifier: result.pipeline_execution?.verifier || "not_reported",
        verifierScope: result.pipeline_execution?.verifier_scope || "not_reported",
        pipeline: result.pipeline_execution || {},
        routing: result.routing_and_planning || {},
        domainPolicy: result.domain_policy || {},
        executionPlan: result.routing_and_planning?.execution_plan,
        responseStatus: result.outcome || result.status,
        requestId: result.request_id,
        processingTimeMs: result.processing_time_ms,
        retryQuestion: prompt,
        onRetry: () => sendQuestion(prompt, { retry: true }),
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

  const handleDocumentUpload = async () => {
    if (
      !uploadDomainId
      || !availableUploadDomains.some((domain) => domain.domain_id === uploadDomainId)
      || !uploadFile
      || uploading
    ) return;
    setUploading(true);
    setUploadStatus("");
    setUploadStatusKind("uploading");
    try {
      const result = await uploadDomainDocument(uploadDomainId, uploadFile, {
        allowedDomainIds: availableUploadDomains.map((domain) => domain.domain_id),
      });
      setUploadStatus(`Đã lập chỉ mục ${result.title || uploadFile.name} · ${result.chunk_count} đoạn · ${result.domain_id}`);
      setUploadStatusKind("success");
      setUploadFile(null);
    } catch (error) {
      setUploadStatus(error.telemetry?.http_response_detail || error.message || "Không thể lập chỉ mục tài liệu.");
      setUploadStatusKind("error");
    } finally {
      setUploading(false);
    }
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
  const availableUploadDomains = uploadRegistrySource === "fallback"
    ? uploadDomains
    : getAvailableUploadDomains(uploadDomains);
  const chatDomainOptions = getDomainChatOptions(domains, domainReadiness);
  const selectedUploadDomain = availableUploadDomains.find((domain) => domain.domain_id === uploadDomainId);
  const supportedUploadExtensions = selectedUploadDomain?.allowed_extensions || [];
  const uploadAccept = uploadAcceptValue(supportedUploadExtensions);
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
<div className="composer-wrap">
              <section className="domain-tools-section" aria-label="Công cụ tài liệu và domain">
                <button
                  className="domain-tools-toggle"
                  type="button"
                  aria-expanded={domainToolsOpen}
                  aria-controls="domain-tools-panel"
                  onClick={() => setDomainToolsOpen((open) => !open)}
                >
                  <span>Công cụ tài liệu và domain</span>
                  {domainToolsOpen
                    ? <ChevronUp size={16} aria-hidden="true" />
                    : <ChevronDown size={16} aria-hidden="true" />}
                </button>
                <div
                  id="domain-tools-panel"
                  className={`domain-tools-panel${domainToolsOpen ? " domain-tools-panel--open" : ""}`}
                  aria-hidden={!domainToolsOpen}
                  inert={!domainToolsOpen}
                >
                  <div className="domain-tools-panel-inner">
                    <div className="domain-tools">
                      <h2 className="domain-upload-title">Nạp tài liệu vào domain</h2>
                      <p className="domain-upload-guide">Chọn domain trước khi tải tài liệu lên</p>
                      <label className="domain-upload-select">
                        <span>Domain đích</span>
                        <select
                          value={uploadDomainId}
                          onChange={(event) => {
                            setUploadDomainId(event.target.value);
                            setUploadFile(null);
                            setUploadStatus("");
                            setUploadStatusKind("ready");
                          }}
                        >
                          <option value="">Chọn domain đích</option>
                          {availableUploadDomains.map((domain) => {
                            const label = domainOptionLabel(domain.display_name_vi || domain.display_name, domain.domain_id);
                            return <option key={domain.domain_id} value={domain.domain_id}>{label}</option>;
                          })}
                        </select>
                      </label>
                      <p className="domain-readiness domain-upload-readiness" role="status">
                        {uploadRegistryLoading
                          ? "Đang tải domain upload từ Registry…"
                          : uploadRegistrySource === "fallback"
                            ? `Danh sách dự phòng đang dùng vì Registry lỗi: ${uploadRegistryErrorMessage} Backend vẫn kiểm tra domain khi nhận upload.`
                            : uploadDomainId
                              ? `Registry cho phép upload. Readiness: ${selectedUploadDomain?.e2e_verified || "not_verified"}; chỉ báo đã xác minh khi API xác nhận đầy đủ.`
                              : availableUploadDomains.length
                                ? "Chọn domain đích. Dữ liệu và E2E có thể vẫn chưa được xác minh; điều đó không ngăn upload được Registry cho phép."
                                : "Registry hiện không công bố domain upload phù hợp."}
                      </p>
                      {uploadRegistryError && (
                        <button
                          type="button"
                          className="domain-upload-retry"
                          onClick={loadUploadDomains}
                          disabled={uploadRegistryLoading}
                        >
                          {uploadRegistryLoading ? "Đang tải…" : "Tải lại danh sách domain"}
                        </button>
                      )}
                      {uploadRegistryError && uploadRegistryDiagnostic && (
                        <details className="domain-registry-diagnostic">
                          <summary>Registry request details</summary>
                          <code>
                            {uploadRegistryDiagnostic.http_method || "GET"} {uploadRegistryDiagnostic.http_path || "/api/v2/domains/upload-targets"}
                            {uploadRegistryDiagnostic.http_status ? ` · HTTP ${uploadRegistryDiagnostic.http_status}` : ""}
                            {uploadRegistryDiagnostic.http_response_detail ? ` · ${uploadRegistryDiagnostic.http_response_detail}` : ""}
                          </code>
                        </details>
                      )}
                      <label
                        className={`domain-upload domain-upload-file domain-upload-dropzone${fileDropActive ? " domain-upload-dropzone--active" : ""}`}
                        onDragEnter={(event) => { event.preventDefault(); setFileDropActive(true); }}
                        onDragOver={(event) => { event.preventDefault(); setFileDropActive(true); }}
                        onDragLeave={(event) => {
                          if (!event.currentTarget.contains(event.relatedTarget)) setFileDropActive(false);
                        }}
                        onDrop={(event) => {
                          event.preventDefault();
                          setFileDropActive(false);
                          const file = event.dataTransfer.files?.[0];
                          if (!file) return;
                          const extension = `.${file.name.split(".").pop()?.toLowerCase() || ""}`;
                          if (!supportedUploadExtensions.includes(extension)) {
                            setUploadStatus(`Định dạng ${extension} không được Registry cho phép.`);
                            setUploadStatusKind("error");
                            return;
                          }
                          setUploadFile(file);
                          setUploadStatus("");
                          setUploadStatusKind("ready");
                        }}
                      >
                        <span>Tệp tài liệu ({supportedUploadExtensions.join(" / ") || "định dạng do Registry công bố"})</span>
                        <input
                          type="file"
                          className="domain-upload-file-input"
                          accept={uploadAccept}
                          disabled={
                            uploadRegistryLoading
                            || !selectedUploadDomain
                            || !supportedUploadExtensions.length
                            || uploading
                          }
                          onChange={(event) => {
                            setUploadFile(event.target.files?.[0] || null);
                            setUploadStatus("");
                            setUploadStatusKind("ready");
                          }}
                        />
                        <span className="domain-upload-dropzone-prompt">
                          <Upload size={17} aria-hidden="true" />
                          {uploadFile ? uploadFile.name : "Chọn tệp hoặc kéo thả vào đây"}
                        </span>
                        <span className="domain-upload-filename">Hỗ trợ {supportedUploadExtensions.join(", ")}</span>
                      </label>
                      <button
                        type="button"
                        className="domain-upload-button"
                        disabled={
                          uploadRegistryLoading
                          || !selectedUploadDomain
                          || !supportedUploadExtensions.length
                          || !uploadFile
                          || uploading
                        }
                        onClick={handleDocumentUpload}
                      >
                        {uploading ? "Đang tải và lập chỉ mục…" : "Tải tài liệu lên domain"}
                      </button>
                      <span className={`domain-upload-status domain-upload-status--${uploadStatusKind}`} role="status">
                        {uploading ? "Đang tải và lập chỉ mục…" : uploadStatus || (uploadFile ? "Sẵn sàng tải lên" : "Chọn domain và tài liệu để bắt đầu")}
                      </span>
                    </div>
                  </div>
                </div>
              </section>
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
                  <label className="chat-domain-control response-style-control">
                    <span>Answer style</span>
                    <select
                      value={responseStyle}
                      onChange={(event) => setResponseStyle(event.target.value)}
                      aria-label="Answer style"
                      disabled={loading}
                    >
                      <option value="default">Mặc định</option>
                      <option value="strict">Strict</option>
                      <option value="friendly">Friendly</option>
                    </select>
                  </label>
                  <label className="chat-domain-control">
                    <span>Domain chat</span>
                    <select value={domainHint} onChange={(event) => setDomainHint(event.target.value)} aria-label="Domain chat" disabled={loading}>
                      <option value="">Automatic (domain routing)</option>
                      {chatDomainOptions.map(({ domainId, displayName }) => {
                        const label = domainOptionLabel(displayName, domainId);
                        return <option key={domainId} value={domainId}>{label}</option>;
                      })}
                    </select>
                  </label>
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
