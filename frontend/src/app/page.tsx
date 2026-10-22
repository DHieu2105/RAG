"use client";

import { FormEvent, useEffect, useRef, useState } from "react";
import Image from "next/image";

const ICONS = {
  bookCard: "/icons/book-card.svg",
  bookCitation: "/icons/book-citation.svg",
  bookTitle: "/icons/book-title.svg",
  brandDot: "/icons/brand-dot.svg",
  chat: "/icons/chat.svg",
  chatActive: "/icons/chat-active.svg",
  copy: "/icons/copy.svg",
  help: "/icons/help.svg",
  online: "/icons/online-dot.svg",
  plus: "/icons/plus.svg",
  send: "/icons/send.svg",
  spark: "/icons/spark.svg",
  sparkLarge: "/icons/spark-large.svg",
  trash: "/icons/trash.svg",
} as const;

const ICON_SIZES: Record<string, [number, number]> = {
  [ICONS.bookCard]: [22, 22],
  [ICONS.bookCitation]: [16, 16],
  [ICONS.bookTitle]: [18, 18],
  [ICONS.brandDot]: [7, 7],
  [ICONS.chat]: [16, 16],
  [ICONS.chatActive]: [16, 16],
  [ICONS.copy]: [14, 14],
  [ICONS.help]: [18, 18],
  [ICONS.online]: [8, 8],
  [ICONS.plus]: [18, 18],
  [ICONS.send]: [20, 20],
  [ICONS.spark]: [13, 13],
  [ICONS.sparkLarge]: [21, 21],
  [ICONS.trash]: [15, 15],
};

const suggestions = ["Đăng ký tín chỉ", "Cảnh báo học tập", "Xét tốt nghiệp"];
const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000/api/v1/chat";
const CONVERSATIONS_STORAGE_KEY = "uniguide-conversations";
const ACTIVE_CONVERSATION_STORAGE_KEY = "uniguide-active-conversation";

type Conversation = {
  id: string;
  title: string;
  messages: Message[];
};

type Message = {
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
};

type Source = {
  part: string | null;
  chapter: string | null;
  article: number | null;
  part_title: string | null;
  chapter_title: string | null;
  article_title: string | null;
  clauses: number[];
  page_start: number | null;
  page_end: number | null;
  claims?: string[];
  evidence?: string[];
};

const defaultConversations: Conversation[] = [];

function loadConversations(): Conversation[] {
  if (typeof window === "undefined") return defaultConversations;

  const savedConversations = window.localStorage.getItem(CONVERSATIONS_STORAGE_KEY);
  if (!savedConversations) return defaultConversations;

  try {
    const parsedConversations = JSON.parse(savedConversations) as Conversation[];
    return Array.isArray(parsedConversations) && parsedConversations.every((conversation) => (
      typeof conversation.id === "string"
      && typeof conversation.title === "string"
      && Array.isArray(conversation.messages)
    )) ? parsedConversations : defaultConversations;
  } catch {
    window.localStorage.removeItem(CONVERSATIONS_STORAGE_KEY);
    return defaultConversations;
  }
}

function loadActiveConversationId(): string | null {
  if (typeof window === "undefined") return null;

  const savedActiveConversationId = window.localStorage.getItem(ACTIVE_CONVERSATION_STORAGE_KEY);
  return savedActiveConversationId;
}

function Icon({ src, alt = "" }: { src: string; alt?: string }) {
  const [width, height] = ICON_SIZES[src] ?? [16, 16];
  return <Image className="ug-icon" src={src} alt={alt} width={width} height={height} unoptimized />;
}

function Sidebar({
  open,
  conversations,
  activeConversationId,
  onNewChat,
  onSelectConversation,
  onDeleteConversation,
  onClose,
}: {
  open: boolean;
  conversations: Conversation[];
  activeConversationId: string | null;
  onNewChat: () => void;
  onSelectConversation: (id: string) => void;
  onDeleteConversation: (id: string) => void;
  onClose: () => void;
}) {
  return (
    <>
      <button className={`sidebar-backdrop ${open ? "is-open" : ""}`} aria-label="Đóng menu" onClick={onClose} />
      <aside className={`sidebar ${open ? "is-open" : ""}`} aria-label="Điều hướng">
        <div className="brand">
          <div className="brand-logo">U</div>
          <div className="brand-copy"><strong>UniGuide AI</strong><span>TRỢ LÝ SINH VIÊN UTC</span></div>
        </div>
        <button className="new-chat" type="button" onClick={onNewChat}><Icon src={ICONS.plus} /><span>Cuộc trò chuyện mới</span></button>
        <p className="sidebar-label">LỊCH SỬ TRÒ CHUYỆN</p>
        <nav className="history-list" aria-label="Lịch sử trò chuyện">
          {conversations.map((conversation) => (
            <div
              className={`history-row ${conversation.id === activeConversationId ? "active" : ""}`}
              key={conversation.id}
            >
              <button
                className="history-item"
                type="button"
                onClick={() => onSelectConversation(conversation.id)}
              >
                <Icon src={conversation.id === activeConversationId ? ICONS.chatActive : ICONS.chat} />
                <span>{conversation.title}</span>
              </button>
              <button
                className="delete-history"
                type="button"
                aria-label={`Xóa cuộc trò chuyện: ${conversation.title}`}
                title="Xóa cuộc trò chuyện"
                onClick={() => {
                  if (window.confirm(`Xóa cuộc trò chuyện “${conversation.title}”?`)) {
                    onDeleteConversation(conversation.id);
                  }
                }}
              >
                <Icon src={ICONS.trash} />
              </button>
            </div>
          ))}
        </nav>
      </aside>
    </>
  );
}

function HelpMenu({ onNewChat }: { onNewChat: () => void }) {
  const [open, setOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;

    function closeOnOutsideClick(event: MouseEvent) {
      if (event.target instanceof Node && !menuRef.current?.contains(event.target)) setOpen(false);
    }

    function closeOnEscape(event: KeyboardEvent) {
      if (event.key === "Escape") setOpen(false);
    }

    document.addEventListener("mousedown", closeOnOutsideClick);
    document.addEventListener("keydown", closeOnEscape);
    return () => {
      document.removeEventListener("mousedown", closeOnOutsideClick);
      document.removeEventListener("keydown", closeOnEscape);
    };
  }, [open]);

  return (
    <div className="help-menu" ref={menuRef}>
      <button
        className="help-button"
        type="button"
        aria-label={open ? "Đóng trợ giúp" : "Mở trợ giúp"}
        aria-expanded={open}
        aria-controls="help-popover"
        onClick={() => setOpen((current) => !current)}
      >
        <Icon src={ICONS.help} />
      </button>
      {open && (
        <div className="help-popover" id="help-popover" role="dialog" aria-label="Hướng dẫn sử dụng">
          <div className="help-popover-title">
            <div className="help-popover-icon"><Icon src={ICONS.help} /></div>
            <div><strong>Trợ giúp nhanh</strong><span>Cách sử dụng UniGuide AI</span></div>
          </div>
          <p>Hãy nhập câu hỏi cụ thể về học vụ, quy chế hoặc quyền lợi sinh viên.</p>
          <div className="help-examples">
            <span>Ví dụ câu hỏi</span>
            <p>“Điều kiện đăng ký tín chỉ là gì?”</p>
            <p>“Khi nào sinh viên bị cảnh báo học tập?”</p>
          </div>
          <button
            className="help-new-chat"
            type="button"
            onClick={() => {
              setOpen(false);
              onNewChat();
            }}
          >
            <Icon src={ICONS.plus} />
            <span>Bắt đầu câu hỏi mới</span>
          </button>
        </div>
      )}
    </div>
  );
}

function Topbar({ onOpenMenu, onNewChat }: { onOpenMenu: () => void; onNewChat: () => void }) {
  return (
    <header className="topbar">
      <button className="mobile-menu" type="button" aria-label="Mở menu" onClick={onOpenMenu}><span /><span /><span /></button>
      <div className="mobile-logo">U</div>
      <div className="mobile-brand"><strong>UniGuide AI</strong><span>TRỢ LÝ SINH VIÊN UTC</span></div>
      <div className="breadcrumbs"><strong>Trợ lý học vụ</strong></div>
      <div className="topbar-spacer" />
      <div className="scope-pill"><Icon src={ICONS.online} /><span>Sổ tay sinh viên K67 Online</span></div>
      <HelpMenu onNewChat={onNewChat} />
      <button className="mobile-new-chat" type="button" aria-label="Cuộc trò chuyện mới" onClick={onNewChat}><Icon src={ICONS.plus} /></button>
    </header>
  );
}

function AssistantAnswer({ answer, sources = [] }: { answer: string; sources?: Source[] }) {
  const [copied, setCopied] = useState(false);

  async function copyAnswer() {
    await navigator.clipboard.writeText(answer);
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1500);
  }

  return (
    <div className="assistant-message dynamic-answer">
      <div className="assistant-avatar"><Icon src={ICONS.sparkLarge} /></div>
      <article className="answer-card">
        <div className="answer-title">
          <h2>Câu trả lời từ UniGuide AI</h2><span>{sources.length ? "Có nguồn dẫn" : "Không có nguồn dẫn"}</span>
        </div>
        <p className="dynamic-answer-text">{answer}</p>
        {sources.length > 0 && (
          <div className="citation">
            <Icon src={ICONS.bookCitation} />
            <span>{sources.map((source) => {
              const clauses = source.clauses.length ? `, Khoản ${source.clauses.join(", ")}` : "";
              const structure = [
                source.part ? `Phần ${source.part}` : "",
                source.chapter ? `Chương ${source.chapter}` : "",
                source.article === null ? "" : `Điều ${source.article}`,
              ].filter(Boolean).join(", ");
              const page = source.page_start === null
                ? ""
                : source.page_end !== null && source.page_end !== source.page_start
                  ? `Trang ${source.page_start}-${source.page_end}`
                  : `Trang ${source.page_start}`;
              return [structure, clauses.replace(/^, /, ""), page].filter(Boolean).join(", ");
            }).join("  |  ")}</span>
          </div>
        )}
        <div className="answer-actions">
          <button type="button" onClick={copyAnswer}><Icon src={ICONS.copy} /><span>{copied ? "Đã sao chép" : "Sao chép"}</span></button>
          <span className="ai-note">Câu trả lời được tạo bởi AI, hãy đối chiếu quy định gốc.</span>
        </div>
      </article>
    </div>
  );
}

function SourcePanel({ sources }: { sources: Source[] }) {
  return (
    <aside className="source-panel" aria-label="Nguồn tham khảo">
      <div className="source-heading"><Icon src={ICONS.bookTitle} /><h2>Nguồn tham khảo</h2></div>
      <p className="source-subtitle">Tài liệu được sử dụng để tạo câu trả lời.</p>
      <div className="document-card">
        <div className="document-icon"><Icon src={ICONS.bookCard} /></div>
        <div className="document-copy">
          <h3>Sổ tay sinh viên K67</h3>
          <p>Trường Đại học Giao thông Vận tải</p>
        </div>
        <div className="document-meta"><strong>PDF · 97 trang</strong><span>Nguồn nội bộ</span></div>
      </div>
      {sources.length > 0 && (
        <>
          <div className="source-section-heading">
            <p className="source-label">ĐOẠN TÀI LIỆU LIÊN QUAN</p>
            <span>{sources.length} nguồn</span>
          </div>
          <div className="citation-card">
            {sources.map((source, index) => {
              const articleName = source.article_title
                ?? (source.article === null ? "Nguồn tài liệu" : `Điều ${source.article}`);
              const page = source.page_start === null
                ? "Không rõ trang"
                : source.page_end !== null && source.page_end !== source.page_start
                  ? `Trang ${source.page_start}-${source.page_end}`
                  : `Trang ${source.page_start}`;
              const clauses = source.clauses?.length ? `Khoản ${source.clauses.join(", ")}` : null;
              const evidence = source.evidence?.filter(Boolean) ?? [];

              return (
                <details className="citation-item" key={`${source.article}-${source.page_start}-${index}`}>
                  <summary>
                    <b>{String(index + 1).padStart(2, "0")}</b>
                    <span className="citation-summary-copy">
                      <strong>{articleName}</strong>
                      <small>{[source.part ? `Phần ${source.part}` : null, clauses, page].filter(Boolean).join(" · ")}</small>
                    </span>
                    <span className="citation-toggle" aria-hidden="true">⌄</span>
                  </summary>
                  <div className="citation-detail">
                    {(source.chapter_title || clauses) && (
                      <p className="citation-context">
                        {[source.chapter_title, clauses].filter(Boolean).join(" · ")}
                      </p>
                    )}
                    {evidence.length > 0 ? evidence.map((excerpt, excerptIndex) => (
                      <blockquote key={`${index}-${excerptIndex}`}>{excerpt}</blockquote>
                    )) : (
                      <p className="citation-empty">Chưa có đoạn trích chi tiết cho nguồn này.</p>
                    )}
                  </div>
                </details>
              );
            })}
          </div>
        </>
      )}
    </aside>
  );
}

export default function Home() {
  const [menuOpen, setMenuOpen] = useState(false);
  const [question, setQuestion] = useState("");
  const [activeConversationId, setActiveConversationId] = useState<string | null>(null);
  const [conversations, setConversations] = useState<Conversation[]>(defaultConversations);
  const [storageReady, setStorageReady] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const requestId = useRef(0);

  useEffect(() => {
    const timer = window.setTimeout(() => {
      const savedConversations = loadConversations();
      const savedActiveConversationId = loadActiveConversationId();
      setConversations(savedConversations);
      setActiveConversationId(
        savedActiveConversationId && savedConversations.some((conversation) => conversation.id === savedActiveConversationId)
          ? savedActiveConversationId
          : null,
      );
      setStorageReady(true);
    }, 0);

    return () => window.clearTimeout(timer);
  }, []);

  useEffect(() => {
    if (!storageReady) return;

    window.localStorage.setItem(CONVERSATIONS_STORAGE_KEY, JSON.stringify(conversations));
    if (activeConversationId === null) {
      window.localStorage.removeItem(ACTIVE_CONVERSATION_STORAGE_KEY);
    } else {
      window.localStorage.setItem(ACTIVE_CONVERSATION_STORAGE_KEY, String(activeConversationId));
    }
  }, [conversations, activeConversationId, storageReady]);

  function startNewChat() {
    requestId.current += 1;
    setActiveConversationId(null);
    setQuestion("");
    setError(null);
    setLoading(false);
    setMenuOpen(false);
  }

  function selectConversation(id: string) {
    requestId.current += 1;
    setActiveConversationId(id);
    setQuestion("");
    setError(null);
    setLoading(false);
    setMenuOpen(false);
  }

  function deleteConversation(id: string) {
    setConversations((current) => current.filter((conversation) => conversation.id !== id));
    if (activeConversationId !== id) return;

    requestId.current += 1;
    setActiveConversationId(null);
    setQuestion("");
    setError(null);
    setLoading(false);
    setMenuOpen(false);
  }

  async function submitQuestion(event?: FormEvent) {
    event?.preventDefault();
    const value = question.trim();
    if (!value || loading) return;
    setQuestion("");
    setError(null);
    setLoading(true);
    const currentRequestId = ++requestId.current;
    const conversationId = activeConversationId ?? crypto.randomUUID();
    setActiveConversationId(conversationId);
    const conversation = conversations.find((item) => item.id === conversationId);
    const history = conversation?.messages ?? [];
    setConversations((current) => {
      if (current.some((item) => item.id === conversationId)) {
        return current.map((item) => item.id === conversationId
          ? { ...item, messages: [...item.messages, { role: "user", content: value }] }
          : item);
      }
      return [...current, { id: conversationId, title: value, messages: [{ role: "user", content: value }] }];
    });
    try {
      const response = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: value, history }),
      });
      if (!response.ok) throw new Error("Request failed");
      const payload = (await response.json()) as { answer: string; sources?: Source[] };
      setConversations((current) => current.map((item) => (
        item.id === conversationId ? {
          ...item,
          messages: [...item.messages, { role: "assistant", content: payload.answer, sources: payload.sources ?? [] }],
        } : item
      )));
    } catch {
      if (requestId.current === currentRequestId) {
        setError("Không thể kết nối tới máy chủ. Hãy kiểm tra backend và thử lại.");
      }
    } finally {
      if (requestId.current === currentRequestId) setLoading(false);
    }
  }

  return (
    <main className="app-shell">
      <Sidebar
        open={menuOpen}
        conversations={conversations}
        activeConversationId={activeConversationId}
        onNewChat={startNewChat}
        onSelectConversation={selectConversation}
        onDeleteConversation={deleteConversation}
        onClose={() => setMenuOpen(false)}
      />
      <section className="main-app">
        <Topbar onOpenMenu={() => setMenuOpen(true)} onNewChat={startNewChat} />
        <div className="workspace">
          <section className="conversation">
            <div className="conversation-scroll">
              {activeConversationId === null ? (
                <>
                  <div className="eyebrow"><Icon src={ICONS.brandDot} /><b>HỎI ĐÁP THÔNG MINH • SỔ TAY K67</b></div>
                  <h1>Chào bạn, mình là UniGuide AI</h1>
                  <p className="intro">Tra cứu quy chế, học vụ và quyền lợi sinh viên UTC một cách nhanh chóng.</p>
                  <div className="suggestions">
                    {suggestions.map((suggestion) => (
                      <button type="button" key={suggestion} onClick={() => setQuestion(suggestion)}><Icon src={ICONS.spark} /><span>{suggestion}</span></button>
                    ))}
                  </div>
                </>
              ) : (
                (() => {
                  const conversation = conversations.find((item) => item.id === activeConversationId);
                  if (!conversation) return null;
                  return (
                    <>
                      <div className="date-divider">HÔM NAY</div>
                      {conversation.messages.map((message, index) => message.role === "user" ? (
                        <div className="user-message" key={`${conversation.id}-${index}`}><p>{message.content}</p></div>
                      ) : (
                        <AssistantAnswer key={`${conversation.id}-${index}`} answer={message.content} sources={message.sources} />
                      ))}
                      {loading && <div className="loading-message">UniGuide AI đang tra cứu tài liệu…</div>}
                      {error && <div className="error-message">{error}</div>}
                    </>
                  );
                })()
              )}
            </div>
            <form className="composer-area" onSubmit={submitQuestion}>
              <div className="composer">
                <input aria-label="Câu hỏi cho UniGuide AI" placeholder="Nhập câu hỏi về học vụ, quy chế, tín chỉ..." value={question} onChange={(event) => setQuestion(event.target.value)} />
                <button type="submit" aria-label="Gửi câu hỏi" disabled={!question.trim() || loading}><Icon src={ICONS.send} /></button>
              </div>
              <p>UniGuide AI sử dụng nội dung Sổ tay K67 để tham khảo, không thay thế thông báo chính thức của Nhà trường.</p>
            </form>
          </section>
          <SourcePanel sources={(() => {
            const conversation = conversations.find((item) => item.id === activeConversationId);
            const lastAssistant = conversation?.messages.slice().reverse().find((message) => message.role === "assistant");
            return lastAssistant?.sources ?? [];
          })()} />
        </div>
      </section>
    </main>
  );
}
