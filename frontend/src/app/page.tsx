"use client";

import { FormEvent, useEffect, useState } from "react";
import Image from "next/image";

const ICONS = {
  bookCard: "/icons/book-card.svg",
  bookCitation: "/icons/book-citation.svg",
  bookTitle: "/icons/book-title.svg",
  brandDot: "/icons/brand-dot.svg",
  chat: "/icons/chat.svg",
  chatActive: "/icons/chat-active.svg",
  chevron: "/icons/chevron.svg",
  copy: "/icons/copy.svg",
  help: "/icons/help.svg",
  more: "/icons/more.svg",
  online: "/icons/online-dot.svg",
  plus: "/icons/plus.svg",
  send: "/icons/send.svg",
  shieldBlue: "/icons/shield-blue.svg",
  shieldLight: "/icons/shield-light.svg",
  spark: "/icons/spark.svg",
  sparkLarge: "/icons/spark-large.svg",
} as const;

const ICON_SIZES: Record<string, [number, number]> = {
  [ICONS.bookCard]: [22, 22],
  [ICONS.bookCitation]: [16, 16],
  [ICONS.bookTitle]: [18, 18],
  [ICONS.brandDot]: [7, 7],
  [ICONS.chat]: [16, 16],
  [ICONS.chatActive]: [16, 16],
  [ICONS.chevron]: [16, 16],
  [ICONS.copy]: [14, 14],
  [ICONS.help]: [18, 18],
  [ICONS.more]: [16, 16],
  [ICONS.online]: [8, 8],
  [ICONS.plus]: [18, 18],
  [ICONS.send]: [20, 20],
  [ICONS.shieldBlue]: [17, 17],
  [ICONS.shieldLight]: [17, 17],
  [ICONS.spark]: [13, 13],
  [ICONS.sparkLarge]: [21, 21],
};

const suggestions = ["Đăng ký tín chỉ", "Cảnh báo học tập", "Xét tốt nghiệp"];
const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000/api/v1/chat";
const CONVERSATIONS_STORAGE_KEY = "uniguide-conversations";
const ACTIVE_CONVERSATION_STORAGE_KEY = "uniguide-active-conversation";

type Conversation = {
  id: number;
  title: string;
  question: string;
  answer: string | null;
};

const defaultConversations: Conversation[] = [
  { id: 0, title: "Cảnh báo kết quả học tập", question: "Sinh viên bị cảnh báo học tập khi nào?", answer: null },
];

function loadConversations(): Conversation[] {
  if (typeof window === "undefined") return defaultConversations;

  const savedConversations = window.localStorage.getItem(CONVERSATIONS_STORAGE_KEY);
  if (!savedConversations) return defaultConversations;

  try {
    const parsedConversations = JSON.parse(savedConversations) as Conversation[];
    return Array.isArray(parsedConversations) ? parsedConversations : defaultConversations;
  } catch {
    window.localStorage.removeItem(CONVERSATIONS_STORAGE_KEY);
    return defaultConversations;
  }
}

function loadActiveConversationId(): number | null {
  if (typeof window === "undefined") return 0;

  const savedActiveConversationId = window.localStorage.getItem(ACTIVE_CONVERSATION_STORAGE_KEY);
  if (savedActiveConversationId === null) return 0;

  const parsedId = Number(savedActiveConversationId);
  return Number.isNaN(parsedId) ? null : parsedId;
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
  onClose,
}: {
  open: boolean;
  conversations: Conversation[];
  activeConversationId: number | null;
  onNewChat: () => void;
  onSelectConversation: (id: number) => void;
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
            <button
              className={`history-item ${conversation.id === activeConversationId ? "active" : ""}`}
              type="button"
              key={conversation.id}
              onClick={() => onSelectConversation(conversation.id)}
            >
              <Icon src={conversation.id === activeConversationId ? ICONS.chatActive : ICONS.chat} />
              <span>{conversation.title}</span>
            </button>
          ))}
        </nav>
        <div className="sidebar-spacer" />
        <div className="verified-box">
          <div><Icon src={ICONS.shieldLight} /><strong>Dữ liệu có kiểm chứng</strong></div>
          <p>Nội dung được tra cứu từ<br />Sổ tay sinh viên K67 (UTC).</p>
        </div>
        <div className="profile">
          <div className="profile-avatar">SV</div>
          <div className="profile-copy"><strong>Sinh viên UTC</strong><span>K67 · Đại học GTVT</span></div>
          <Icon src={ICONS.more} />
        </div>
      </aside>
    </>
  );
}

function Topbar({ onOpenMenu, onNewChat }: { onOpenMenu: () => void; onNewChat: () => void }) {
  return (
    <header className="topbar">
      <button className="mobile-menu" type="button" aria-label="Mở menu" onClick={onOpenMenu}><span /><span /><span /></button>
      <div className="mobile-logo">U</div>
      <div className="mobile-brand"><strong>UniGuide AI</strong><span>TRỢ LÝ SINH VIÊN UTC</span></div>
      <div className="breadcrumbs"><span>TRANG CHỦ</span><Icon src={ICONS.chevron} /><strong>Trợ lý học vụ</strong></div>
      <div className="topbar-spacer" />
      <div className="scope-pill"><Icon src={ICONS.online} /><span>Sổ tay sinh viên K67</span></div>
      <button className="help-button" type="button" aria-label="Trợ giúp"><Icon src={ICONS.help} /></button>
      <button className="mobile-new-chat" type="button" aria-label="Cuộc trò chuyện mới" onClick={onNewChat}><Icon src={ICONS.plus} /></button>
    </header>
  );
}

function AssistantAnswer({ answer, dynamic = false }: { answer?: string; dynamic?: boolean }) {
  const [copied, setCopied] = useState(false);
  const copyText = answer ?? "Theo Điều 11, sinh viên được cảnh báo kết quả học tập sau mỗi học kỳ chính khi vi phạm một trong các trường hợp theo quy định.";

  async function copyAnswer() {
    await navigator.clipboard.writeText(copyText);
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1500);
  }

  return (
    <div className={`assistant-message ${dynamic ? "dynamic-answer" : ""}`}>
      <div className="assistant-avatar"><Icon src={ICONS.sparkLarge} /></div>
      <article className="answer-card">
        <div className="answer-title">
          <h2>{dynamic ? "Câu trả lời từ UniGuide AI" : "Cảnh báo kết quả học tập"}</h2><span>Có nguồn dẫn</span>
        </div>
        {dynamic ? <p className="dynamic-answer-text">{answer}</p> : (
          <>
            <p className="answer-summary">Theo Điều 11, sinh viên được cảnh báo kết quả học tập sau mỗi học kỳ chính khi vi phạm một trong các trường hợp sau:</p>
            <div className="conditions">
              <p><b>01</b><span>Không đăng ký đủ số tín chỉ tối thiểu theo quy định.</span></p>
              <p><b>02</b><span>Điểm trung bình học kỳ dưới 0,8 ở học kỳ đầu hoặc<br />dưới 1,0 ở các học kỳ tiếp theo.</span></p>
            </div>
          </>
        )}
        <div className="citation"><Icon src={ICONS.bookCitation} /><span>Sổ tay K67&nbsp; · &nbsp;Điều 11, Khoản 1&nbsp; · &nbsp;Trang 23</span></div>
        <div className="answer-actions">
          <button type="button" onClick={copyAnswer}><Icon src={ICONS.copy} /><span>{copied ? "Đã sao chép" : "Sao chép"}</span></button>
          <span className="ai-note">Câu trả lời được tạo bởi AI, hãy đối chiếu quy định gốc.</span>
        </div>
      </article>
    </div>
  );
}

function SourcePanel() {
  return (
    <aside className="source-panel" aria-label="Nguồn tham khảo">
      <div className="source-heading"><Icon src={ICONS.bookTitle} /><h2>Nguồn tham khảo</h2></div>
      <p className="source-subtitle">Tài liệu được sử dụng để tạo câu trả lời.</p>
      <div className="document-card">
        <div className="document-icon"><Icon src={ICONS.bookCard} /></div>
        <h3>Sổ tay sinh viên K67</h3>
        <p>Trường Đại học Giao thông Vận tải<br />Bộ dữ liệu quy chế &amp; học vụ</p>
        <div><strong>PDF · 97 trang</strong><span>Nguồn nội bộ</span></div>
      </div>
      <p className="source-label">ĐOẠN TÀI LIỆU LIÊN QUAN</p>
      <div className="citation-card">
        <div className="citation-card-title"><b>01</b><strong>Điều 11 · Cảnh báo học tập</strong></div>
        <p>Sau mỗi học kỳ chính, sinh viên được cảnh báo kết quả học tập nếu vi phạm...</p>
        <div className="source-tags"><span>Trang 23</span><strong>Khớp cao nhất</strong></div>
      </div>
      <div className="accuracy-card">
        <div><Icon src={ICONS.shieldBlue} /><strong>Minh bạch nguồn dẫn</strong></div>
        <p>Mỗi câu trả lời gắn Điều, Khoản và trang PDF để bạn tự đối chiếu thông tin khi cần.</p>
      </div>
    </aside>
  );
}

export default function Home() {
  const [menuOpen, setMenuOpen] = useState(false);
  const [question, setQuestion] = useState("");
  const [activeConversationId, setActiveConversationId] = useState<number | null>(0);
  const [conversations, setConversations] = useState<Conversation[]>(defaultConversations);
  const [storageReady, setStorageReady] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const timer = window.setTimeout(() => {
      setConversations(loadConversations());
      setActiveConversationId(loadActiveConversationId());
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
    setActiveConversationId(null);
    setQuestion("");
    setError(null);
    setLoading(false);
    setMenuOpen(false);
  }

  function selectConversation(id: number) {
    setActiveConversationId(id);
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
    const conversationId = Date.now();
    setActiveConversationId(conversationId);
    setConversations((current) => [
      ...current,
      { id: conversationId, title: value, question: value, answer: null },
    ]);
    try {
      const response = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: value }),
      });
      if (!response.ok) throw new Error("Request failed");
      const payload = (await response.json()) as { answer: string };
      setConversations((current) => current.map((conversation) => (
        conversation.id === conversationId ? { ...conversation, answer: payload.answer } : conversation
      )));
    } catch {
      setError("Không thể kết nối tới máy chủ. Hãy kiểm tra backend và thử lại.");
    } finally {
      setLoading(false);
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
                      <div className="user-message"><p>{conversation.question}</p></div>
                      {conversation.id === 0 ? <AssistantAnswer /> : (
                        <>
                          {loading && <div className="loading-message">UniGuide AI đang tra cứu tài liệu…</div>}
                          {conversation.answer && <AssistantAnswer answer={conversation.answer} dynamic />}
                          {error && <div className="error-message">{error}</div>}
                        </>
                      )}
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
          <SourcePanel />
        </div>
      </section>
    </main>
  );
}
