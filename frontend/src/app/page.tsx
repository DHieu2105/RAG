"use client";

import { FormEvent, Fragment, MouseEvent as ReactMouseEvent, useEffect, useRef, useState } from "react";
import Image from "next/image";

const APP_NAME = "Unimate";
const LOGO_SRC = "/logo.png";

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
  [ICONS.bookCard]: [20, 20],
  [ICONS.bookCitation]: [15, 15],
  [ICONS.bookTitle]: [17, 17],
  [ICONS.brandDot]: [6, 6],
  [ICONS.chat]: [15, 15],
  [ICONS.chatActive]: [15, 15],
  [ICONS.copy]: [13, 13],
  [ICONS.help]: [17, 17],
  [ICONS.online]: [7, 7],
  [ICONS.plus]: [17, 17],
  [ICONS.send]: [19, 19],
  [ICONS.spark]: [13, 13],
  [ICONS.sparkLarge]: [20, 20],
  [ICONS.trash]: [14, 14],
};

const SUGGESTIONS = [
  {
    category: "Đào tạo & Tín chỉ",
    title: "Đăng ký tín chỉ",
    desc: "Quy định số tín chỉ tối thiểu & tối đa mỗi kỳ",
    query: "Quy định về số tín chỉ tối thiểu và tối đa sinh viên UTC được đăng ký trong một học kỳ?",
    badge: "Học kỳ 1 & 2",
  },
  {
    category: "Quy chế học vụ",
    title: "Cảnh báo học tập",
    desc: "Điều kiện bị cảnh báo học tập & buộc thôi học",
    query: "Sinh viên UTC bị cảnh báo học tập khi nào và bị mấy lần thì buộc thôi học?",
    badge: "Quy chế đào tạo",
  },
  {
    category: "Chuẩn đầu ra",
    title: "Xét tốt nghiệp",
    desc: "Điều kiện tốt nghiệp & chuẩn ngoại ngữ, tin học",
    query: "Điều kiện công nhận tốt nghiệp và chuẩn đầu ra Ngoại ngữ, Tin học của K67 UTC?",
    badge: "Chuẩn K67",
  },
  {
    category: "Chế độ chính sách",
    title: "Học bổng & Rèn luyện",
    desc: "Tiêu chí xét học bổng khuyến khích học tập",
    query: "Tiêu chuẩn xét cấp học bổng khuyến khích học tập và cách đánh giá điểm rèn luyện tại UTC?",
    badge: "Chính sách",
  },
];

const API_URL = "/api/chat";
const CONVERSATIONS_STORAGE_KEY = "utc-assistant-conversations";
const LEGACY_STORAGE_KEY = "uniguide-conversations";
const ACTIVE_CONVERSATION_STORAGE_KEY = "utc-assistant-active-conversation";
const LEGACY_ACTIVE_KEY = "uniguide-active-conversation";

type Conversation = {
  id: string;
  title: string;
  createdAt?: number;
  updatedAt?: number;
  messages: Message[];
};

type Message = {
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
  status?: "sending" | "failed";
  timestamp?: number;
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

function isSameCalendarDay(t1?: number, t2?: number): boolean {
  if (!t1 || !t2) return true;
  const d1 = new Date(t1);
  const d2 = new Date(t2);
  return (
    d1.getFullYear() === d2.getFullYear() &&
    d1.getMonth() === d2.getMonth() &&
    d1.getDate() === d2.getDate()
  );
}

function formatDateLabel(timestamp?: number): string {
  if (!timestamp) return "HÔM NAY";

  const date = new Date(timestamp);
  const now = new Date();

  const startOfToday = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime();
  const startOfYesterday = new Date(now.getFullYear(), now.getMonth(), now.getDate() - 1).getTime();
  const startOfMsgDate = new Date(date.getFullYear(), date.getMonth(), date.getDate()).getTime();

  if (startOfMsgDate === startOfToday) {
    return "HÔM NAY";
  }
  if (startOfMsgDate === startOfYesterday) {
    return "HÔM QUA";
  }

  const dd = String(date.getDate()).padStart(2, "0");
  const mm = String(date.getMonth() + 1).padStart(2, "0");
  const yyyy = date.getFullYear();
  return `${dd}/${mm}/${yyyy}`;
}

const defaultConversations: Conversation[] = [];

function loadConversations(): Conversation[] {
  if (typeof window === "undefined") return defaultConversations;

  const savedConversations = window.localStorage.getItem(CONVERSATIONS_STORAGE_KEY)
    ?? window.localStorage.getItem(LEGACY_STORAGE_KEY);
  if (!savedConversations) return defaultConversations;

  try {
    const parsedConversations = JSON.parse(savedConversations) as Conversation[];
    return Array.isArray(parsedConversations) && parsedConversations.every((conversation) => (
      typeof conversation.id === "string"
      && typeof conversation.title === "string"
      && Array.isArray(conversation.messages)
    )) ? parsedConversations.map((conversation) => ({
      ...conversation,
      createdAt: conversation.createdAt ?? Date.now(),
      messages: conversation.messages.map((message) => (
        message.status === "sending" ? { ...message, status: "failed" } : message
      )),
    })) : defaultConversations;
  } catch {
    window.localStorage.removeItem(CONVERSATIONS_STORAGE_KEY);
    return defaultConversations;
  }
}

function loadActiveConversationId(): string | null {
  if (typeof window === "undefined") return null;
  return window.localStorage.getItem(ACTIVE_CONVERSATION_STORAGE_KEY)
    ?? window.localStorage.getItem(LEGACY_ACTIVE_KEY);
}

function Icon({ src, alt = "" }: { src: string; alt?: string }) {
  const [width, height] = ICON_SIZES[src] ?? [16, 16];
  return <Image className="ug-icon" src={src} alt={alt} width={width} height={height} unoptimized />;
}

function Sidebar({
  open,
  collapsed,
  conversations,
  activeConversationId,
  backendStatus,
  onNewChat,
  onSelectConversation,
  onDeleteConversation,
  onClose,
}: {
  open: boolean;
  collapsed: boolean;
  conversations: Conversation[];
  activeConversationId: string | null;
  backendStatus: "checking" | "online" | "offline";
  onNewChat: () => void;
  onSelectConversation: (id: string) => void;
  onDeleteConversation: (id: string) => void;
  onClose: () => void;
}) {
  return (
    <>
      <button className={`sidebar-backdrop ${open ? "is-open" : ""}`} aria-label="Đóng menu" onClick={onClose} />
      <aside className={`sidebar ${open ? "is-open" : ""} ${collapsed ? "is-collapsed" : ""}`} aria-label="Điều hướng chính">
        <div className="brand">
          <div className="brand-logo-container">
            <Image
              src={LOGO_SRC}
              alt="Logo Trường Đại học Giao thông Vận tải"
              width={38}
              height={38}
              className="utc-logo-img"
              unoptimized
            />
          </div>
          <div className="brand-copy">
            <strong>{APP_NAME}</strong>
            <span>ĐẠI HỌC GIAO THÔNG VẬN TẢI</span>
          </div>
        </div>

        <button className="new-chat" type="button" onClick={onNewChat}>
          <div className="new-chat-icon">
            <Icon src={ICONS.plus} />
          </div>
          <span>Cuộc trò chuyện mới</span>
        </button>

        <div className="sidebar-section-header">
          <span className="sidebar-label">LỊCH SỬ TRA CỨU</span>
          <span className="history-count">{conversations.length}</span>
        </div>

        <nav className="history-list" aria-label="Lịch sử trò chuyện">
          {conversations.length === 0 ? (
            <div className="history-empty">
              <span>Chưa có cuộc trò chuyện nào</span>
            </div>
          ) : (
            conversations.map((conversation) => {
              const isActive = conversation.id === activeConversationId;
              return (
                <div
                  className={`history-row ${isActive ? "active" : ""}`}
                  key={conversation.id}
                >
                  <button
                    className="history-item"
                    type="button"
                    onClick={() => onSelectConversation(conversation.id)}
                  >
                    <Icon src={isActive ? ICONS.chatActive : ICONS.chat} />
                    <span>{conversation.title}</span>
                  </button>
                  <button
                    className="delete-history"
                    type="button"
                    aria-label={`Xóa: ${conversation.title}`}
                    title="Xóa cuộc trò chuyện này"
                    onClick={(e) => {
                      e.stopPropagation();
                      if (window.confirm(`Xóa cuộc trò chuyện “${conversation.title}”?`)) {
                        onDeleteConversation(conversation.id);
                      }
                    }}
                  >
                    <Icon src={ICONS.trash} />
                  </button>
                </div>
              );
            })
          )}
        </nav>

        <div className="sidebar-footer">
          <div className={`utc-status-card is-${backendStatus}`}>
            <div className="utc-badge-pulse" />
            <div className="utc-status-copy">
              <strong>Sổ tay Sinh viên K67</strong>
              <span>
                {backendStatus === "online" && "Đại học Giao thông Vận tải • Online"}
                {backendStatus === "offline" && "Chưa kết nối Backend (Offline)"}
                {backendStatus === "checking" && "Đang kiểm tra kết nối..."}
              </span>
            </div>
          </div>
        </div>
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
        aria-label={open ? "Đóng hướng dẫn" : "Mở hướng dẫn sử dụng"}
        aria-expanded={open}
        aria-controls="help-popover"
        onClick={() => setOpen((current) => !current)}
      >
        <Icon src={ICONS.help} />
      </button>
      {open && (
        <div className="help-popover" id="help-popover" role="dialog" aria-label="Hướng dẫn sử dụng trợ lý UTC">
          <div className="help-popover-title">
            <div className="help-popover-icon">
              <Icon src={ICONS.help} />
            </div>
            <div>
              <strong>Cẩm nang tra cứu học vụ</strong>
              <span>Dành cho sinh viên Trường ĐH Giao thông Vận tải</span>
            </div>
          </div>
          <p className="help-desc">
            {APP_NAME} đối chiếu trực tiếp từ <strong>Sổ tay sinh viên K67</strong> của UTC. Hãy hỏi rõ ràng và cụ thể về:
          </p>
          <div className="help-examples">
            <div className="help-example-pill">Đăng ký tín chỉ & xét vượt</div>
            <div className="help-example-pill">Quy chế thi & điểm rèn luyện</div>
            <div className="help-example-pill">Chuẩn đầu ra Ngoại ngữ & Tin học</div>
            <div className="help-example-pill">Học bổng & miễn giảm học phí</div>
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
            <span>Tạo câu hỏi mới</span>
          </button>
        </div>
      )}
    </div>
  );
}

function ThemeToggle({ theme, onToggle }: { theme: "light" | "dark"; onToggle: () => void }) {
  return (
    <button
      className="theme-toggle-btn"
      type="button"
      onClick={onToggle}
      aria-label={theme === "dark" ? "Chuyển sang chế độ sáng" : "Chuyển sang chế độ tối"}
      title={theme === "dark" ? "Giao diện: Tối (nhấn chuyển Sáng)" : "Giao diện: Sáng (nhấn chuyển Tối)"}
    >
      {theme === "dark" ? (
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
          <circle cx="12" cy="12" r="5" fill="#f59e0b" stroke="#f59e0b" />
          <line x1="12" y1="1" x2="12" y2="3" stroke="#f59e0b" />
          <line x1="12" y1="21" x2="12" y2="23" stroke="#f59e0b" />
          <line x1="4.22" y1="4.22" x2="5.64" y2="5.64" stroke="#f59e0b" />
          <line x1="18.36" y1="18.36" x2="19.78" y2="19.78" stroke="#f59e0b" />
          <line x1="1" y1="12" x2="3" y2="12" stroke="#f59e0b" />
          <line x1="21" y1="12" x2="23" y2="12" stroke="#f59e0b" />
          <line x1="4.22" y1="19.78" x2="5.64" y2="18.36" stroke="#f59e0b" />
          <line x1="18.36" y1="5.64" x2="19.78" y2="4.22" stroke="#f59e0b" />
        </svg>
      ) : (
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
          <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" fill="currentColor" />
        </svg>
      )}
    </button>
  );
}

function Topbar({
  onOpenMenu,
  onNewChat,
  theme,
  onToggleTheme,
  sidebarCollapsed,
  onToggleSidebar,
  sourcePanelOpen,
  onToggleSourcePanel,
  sourceCount,
  backendStatus,
}: {
  onOpenMenu: () => void;
  onNewChat: () => void;
  theme: "light" | "dark";
  onToggleTheme: () => void;
  sidebarCollapsed: boolean;
  onToggleSidebar: () => void;
  sourcePanelOpen: boolean;
  onToggleSourcePanel: () => void;
  sourceCount: number;
  backendStatus: "checking" | "online" | "offline";
}) {
  return (
    <header className="topbar">
      <button className="mobile-menu" type="button" aria-label="Mở danh mục" onClick={onOpenMenu}>
        <span />
        <span />
        <span />
      </button>

      {/* Button to toggle left sidebar */}
      <button
        className={`panel-toggle-btn desktop-sidebar-btn ${sidebarCollapsed ? "is-collapsed-state" : "is-open-state"}`}
        type="button"
        onClick={onToggleSidebar}
        title={sidebarCollapsed ? "Mở rộng thanh bên trái (Lịch sử)" : "Thu gọn thanh bên trái (Lịch sử)"}
        aria-label="Thu gọn/mở thanh bên trái"
      >
        <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <rect x="3" y="3" width="18" height="18" rx="2" ry="2" />
          <line x1="9" y1="3" x2="9" y2="21" />
          {sidebarCollapsed ? <path d="m13 9 3 3-3 3" /> : <path d="m15 9-3 3 3 3" />}
        </svg>
      </button>

      <div className="mobile-brand-wrapper">
        <div className="mobile-logo-container">
          <Image
            src={LOGO_SRC}
            alt="Logo UTC"
            width={28}
            height={28}
            className="utc-logo-img"
            unoptimized
          />
        </div>
        <div className="mobile-brand">
          <strong>{APP_NAME}</strong>
          <span>TRỢ LÝ SINH VIÊN UTC</span>
        </div>
      </div>

      <div className="breadcrumbs">
        <span className="breadcrumb-school">ĐH Giao thông Vận tải</span>
        <span className="breadcrumb-divider">/</span>
        <strong>Cố vấn học vụ & quy chế K67</strong>
      </div>

      <div className="topbar-spacer" />

      <div
        className={`scope-pill is-${backendStatus}`}
        title={
          backendStatus === "online"
            ? "Đã kết nối với máy chủ AI Backend"
            : backendStatus === "offline"
              ? "Chưa kết nối được với máy chủ AI Backend"
              : "Đang kiểm tra kết nối Backend..."
        }
      >
        <span className="scope-dot" />
        <span>
          {backendStatus === "online" && "Sổ tay K67 Online"}
          {backendStatus === "offline" && "Mất kết nối Backend"}
          {backendStatus === "checking" && "Đang kết nối..."}
        </span>
      </div>

      {/* Button to toggle right Source Panel */}
      <button
        className={`panel-toggle-btn source-toggle-btn ${sourcePanelOpen ? "is-active" : ""}`}
        type="button"
        onClick={onToggleSourcePanel}
        title={sourcePanelOpen ? "Thu gọn bảng tài liệu nguồn" : "Mở bảng tài liệu nguồn"}
        aria-label="Thu gọn/mở bảng tài liệu nguồn"
      >
        <Icon src={ICONS.bookCitation} />
        <span className="source-toggle-label">Tài liệu đối chiếu</span>
        {sourceCount > 0 && <span className="source-toggle-badge">{sourceCount}</span>}
      </button>

      <ThemeToggle theme={theme} onToggle={onToggleTheme} />
      <HelpMenu onNewChat={onNewChat} />
      <button className="mobile-new-chat" type="button" aria-label="Cuộc trò chuyện mới" onClick={onNewChat}>
        <Icon src={ICONS.plus} />
      </button>
    </header>
  );
}

function AssistantAnswer({ answer, sources = [] }: { answer: string; sources?: Source[] }) {
  const [copied, setCopied] = useState(false);

  async function copyAnswer() {
    await navigator.clipboard.writeText(answer);
    setCopied(true);
    window.setTimeout(() => setCopied(false), 2000);
  }

  return (
    <div className="assistant-message dynamic-answer">
      <div className="assistant-avatar">
        <Image
          src={LOGO_SRC}
          alt={APP_NAME}
          width={26}
          height={26}
          className="utc-avatar-img"
          unoptimized
        />
      </div>
      <article className="answer-card">
        <div className="answer-title">
          <div className="answer-title-group">
            <h2>{APP_NAME} — Cố vấn học vụ UTC</h2>
            <span className="verified-badge">
              <span className="verified-dot" />
              Sổ tay K67
            </span>
          </div>
          <span className={`source-status-badge ${sources.length > 0 ? "has-sources" : ""}`}>
            {sources.length > 0 ? `${sources.length} trích dẫn` : "Phản hồi AI"}
          </span>
        </div>

        <div className="dynamic-answer-text">
          {answer.split("\n\n").map((paragraph, pIdx) => (
            <p key={pIdx}>{paragraph}</p>
          ))}
        </div>

        {sources.length > 0 && (
          <div className="citation-banner">
            <div className="citation-banner-label">
              <Icon src={ICONS.bookCitation} />
              <span>Nguồn quy định đối chiếu:</span>
            </div>
            <div className="citation-chips">
              {sources.map((source, sIdx) => {
                const clauses = source.clauses.length ? `Khoản ${source.clauses.join(", ")}` : "";
                const structure = [
                  source.part ? `Phần ${source.part}` : "",
                  source.chapter ? `Chương ${source.chapter}` : "",
                  source.article === null ? "" : `Điều ${source.article}`,
                ].filter(Boolean).join(", ");
                const page = source.page_start === null
                  ? ""
                  : source.page_end !== null && source.page_end !== source.page_start
                    ? `Tr. ${source.page_start}-${source.page_end}`
                    : `Tr. ${source.page_start}`;
                const text = [structure, clauses, page].filter(Boolean).join(" · ");
                return (
                  <span className="citation-chip" key={sIdx}>
                    {text}
                  </span>
                );
              })}
            </div>
          </div>
        )}

        <div className="answer-actions">
          <button type="button" className={`copy-btn ${copied ? "is-copied" : ""}`} onClick={copyAnswer}>
            <Icon src={ICONS.copy} />
            <span>{copied ? "Đã sao chép" : "Sao chép"}</span>
          </button>
          <span className="ai-note">
            Thông tin mang tính tham khảo từ Sổ tay K67. Đối chiếu quy định chính thức tại Phòng Đào tạo UTC.
          </span>
        </div>
      </article>
    </div>
  );
}

function SourcePanel({
  sources,
  open,
  onClose,
}: {
  sources: Source[];
  open: boolean;
  onClose: () => void;
}) {
  return (
    <aside className={`source-panel ${open ? "is-open" : "is-collapsed"}`} aria-label="Tài liệu tham khảo chính thức">
      <div className="source-heading">
        <div className="source-heading-icon">
          <Icon src={ICONS.bookTitle} />
        </div>
        <div className="source-heading-text">
          <h2>Tài liệu đối chiếu</h2>
          <p className="source-subtitle">Cơ sở dữ liệu Sổ tay K67 UTC</p>
        </div>
        <button
          className="panel-close-btn"
          type="button"
          onClick={onClose}
          title="Thu gọn bảng tài liệu"
          aria-label="Thu gọn bảng tài liệu"
        >
          ✕
        </button>
      </div>

      <div className="document-card">
        <div className="document-icon">
          <Image
            src={LOGO_SRC}
            alt="Sổ tay K67 UTC"
            width={28}
            height={28}
            className="utc-logo-img"
            unoptimized
          />
        </div>
        <div className="document-copy">
          <h3>Sổ tay Sinh viên K67</h3>
          <p>Trường Đại học Giao thông Vận tải</p>
        </div>
        <div className="document-meta">
          <strong className="document-pages-tag">PDF · 97 trang</strong>
          <span className="document-official-tag">Văn bản chính quy</span>
        </div>
      </div>

      {sources.length > 0 ? (
        <>
          <div className="source-section-heading">
            <p className="source-label">CĂN CỨ TRÍCH DẪN</p>
            <span className="source-count-badge">{sources.length} trích đoạn</span>
          </div>
          <div className="citation-card">
            {sources.map((source, index) => {
              const articleName = source.article_title
                ?? (source.article === null ? "Quy định liên quan" : `Điều ${source.article}`);
              const page = source.page_start === null
                ? "Không rõ trang"
                : source.page_end !== null && source.page_end !== source.page_start
                  ? `Trang ${source.page_start}-${source.page_end}`
                  : `Trang ${source.page_start}`;
              const clauses = source.clauses?.length ? `Khoản ${source.clauses.join(", ")}` : null;
              const evidence = source.evidence?.filter(Boolean) ?? [];

              return (
                <details className="citation-item" key={`${source.article}-${source.page_start}-${index}`} open={index === 0}>
                  <summary>
                    <b className="citation-idx">{String(index + 1).padStart(2, "0")}</b>
                    <span className="citation-summary-copy">
                      <strong>{articleName}</strong>
                      <small>{[source.part ? `Phần ${source.part}` : null, clauses, page].filter(Boolean).join(" · ")}</small>
                    </span>
                    <span className="citation-toggle" aria-hidden="true">⌄</span>
                  </summary>
                  <div className="citation-detail">
                    {(source.chapter_title || clauses) && (
                      <p className="citation-context">
                        {[source.chapter_title, clauses].filter(Boolean).join(" • ")}
                      </p>
                    )}
                    {evidence.length > 0 ? evidence.map((excerpt, excerptIndex) => (
                      <blockquote key={`${index}-${excerptIndex}`}>{excerpt}</blockquote>
                    )) : (
                      <p className="citation-empty">Đoạn trích từ nội dung điều khoản Sổ tay K67.</p>
                    )}
                  </div>
                </details>
              );
            })}
          </div>
        </>
      ) : (
        <div className="source-empty-state">
          <div className="source-empty-icon">📖</div>
          <p>Khi bạn hỏi câu hỏi học vụ, các điều khoản và trang liên quan sẽ xuất hiện tại đây.</p>
        </div>
      )}
    </aside>
  );
}

export default function Home() {
  const [menuOpen, setMenuOpen] = useState(false);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [sourcePanelOpen, setSourcePanelOpen] = useState(true);
  const [question, setQuestion] = useState("");
  const [activeConversationId, setActiveConversationId] = useState<string | null>(null);
  const [conversations, setConversations] = useState<Conversation[]>(defaultConversations);
  const [storageReady, setStorageReady] = useState(false);
  const [loading, setLoading] = useState(false);
  const [theme, setTheme] = useState<"light" | "dark">("light");
  const [backendStatus, setBackendStatus] = useState<"checking" | "online" | "offline">("checking");
  const requestId = useRef(0);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    let active = true;

    async function checkHealth() {
      try {
        const res = await fetch("/api/health", { cache: "no-store" });
        if (active) {
          setBackendStatus(res.ok ? "online" : "offline");
        }
      } catch {
        if (active) {
          setBackendStatus("offline");
        }
      }
    }

    void checkHealth();
    const interval = setInterval(checkHealth, 10000);
    return () => {
      active = false;
      clearInterval(interval);
    };
  }, []);

  useEffect(() => {
    const timer = window.setTimeout(() => {
      const savedTheme = window.localStorage.getItem("utc-assistant-theme") as "light" | "dark" | null;
      const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
      const initialTheme = savedTheme ?? (prefersDark ? "dark" : "light");
      setTheme(initialTheme);
      document.documentElement.setAttribute("data-theme", initialTheme);
    }, 0);

    return () => window.clearTimeout(timer);
  }, []);

  function toggleTheme() {
    const nextTheme = theme === "light" ? "dark" : "light";
    setTheme(nextTheme);
    window.localStorage.setItem("utc-assistant-theme", nextTheme);
    document.documentElement.setAttribute("data-theme", nextTheme);
  }

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

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [conversations, loading]);

  function startNewChat() {
    requestId.current += 1;
    setActiveConversationId(null);
    setQuestion("");
    setLoading(false);
    setMenuOpen(false);
  }

  function selectConversation(id: string) {
    requestId.current += 1;
    setActiveConversationId(id);
    setQuestion("");
    setLoading(false);
    setMenuOpen(false);
  }

  function deleteConversation(id: string) {
    setConversations((current) => current.filter((conversation) => conversation.id !== id));
    if (activeConversationId !== id) return;

    requestId.current += 1;
    setActiveConversationId(null);
    setQuestion("");
    setLoading(false);
    setMenuOpen(false);
  }

  async function sendQuestion(value: string, retryMessageIndex?: number) {
    if (!value || loading) return;
    setLoading(true);
    const currentRequestId = ++requestId.current;
    const conversationId = activeConversationId ?? crypto.randomUUID();
    setActiveConversationId(conversationId);
    const conversation = conversations.find((item) => item.id === conversationId);
    const history = (
      retryMessageIndex === undefined
        ? conversation?.messages ?? []
        : conversation?.messages.slice(0, retryMessageIndex) ?? []
    ).filter((message) => message.status !== "failed");

    const now = Date.now();
    setConversations((current) => {
      if (current.some((item) => item.id === conversationId)) {
        return current.map((item) => item.id === conversationId
          ? {
              ...item,
              updatedAt: now,
              messages: retryMessageIndex === undefined
                ? [...item.messages, { role: "user", content: value, status: "sending", timestamp: now }]
                : item.messages.map((message, index) => (
                    index === retryMessageIndex ? { ...message, status: "sending", timestamp: now } : message
                  )),
            }
          : item);
      }
      return [...current, {
        id: conversationId,
        title: value.length > 40 ? `${value.slice(0, 40)}...` : value,
        createdAt: now,
        updatedAt: now,
        messages: [{ role: "user", content: value, status: "sending", timestamp: now }],
      }];
    });

    try {
      const response = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: value, history }),
      });
      if (!response.ok) throw new Error("Request failed");
      setBackendStatus("online");
      const payload = (await response.json()) as { answer: string; sources?: Source[] };
      const assistantTimestamp = Date.now();
      setConversations((current) => current.map((item) => (
        item.id === conversationId ? {
          ...item,
          updatedAt: assistantTimestamp,
          messages: [
            ...item.messages.map((message, index) => (
              index === (retryMessageIndex ?? item.messages.length - 1)
                ? { ...message, status: undefined }
                : message
            )),
            { role: "assistant", content: payload.answer, sources: payload.sources ?? [], timestamp: assistantTimestamp },
          ],
        } : item
      )));
    } catch {
      setBackendStatus("offline");
      setConversations((current) => current.map((item) => (
        item.id === conversationId ? {
          ...item,
          messages: item.messages.map((message, index) => (
            index === (retryMessageIndex ?? item.messages.length - 1)
              ? { ...message, status: "failed" }
              : message
          )),
        } : item
      )));
    } finally {
      if (requestId.current === currentRequestId) setLoading(false);
    }
  }

  async function submitQuestion(event?: FormEvent) {
    event?.preventDefault();
    const value = question.trim();
    if (!value || loading) return;
    setQuestion("");
    await sendQuestion(value);
  }

  function handleQuickPrompt(promptQuery: string) {
    setQuestion(promptQuery);
  }

  function retryQuestion(event: ReactMouseEvent<HTMLButtonElement>) {
    const messageIndex = Number(event.currentTarget.dataset.messageIndex);
    const conversation = conversations.find((item) => item.id === activeConversationId);
    const message = conversation?.messages[messageIndex];
    if (!message || message.role !== "user" || message.status !== "failed") return;
    void sendQuestion(message.content, messageIndex);
  }

  const activeConversation = conversations.find((item) => item.id === activeConversationId);
  const activeSources = (() => {
    const lastAssistant = activeConversation?.messages.slice().reverse().find((message) => message.role === "assistant");
    return lastAssistant?.sources ?? [];
  })();

  return (
    <main className="app-shell">
      <Sidebar
        open={menuOpen}
        collapsed={sidebarCollapsed}
        conversations={conversations}
        activeConversationId={activeConversationId}
        backendStatus={backendStatus}
        onNewChat={startNewChat}
        onSelectConversation={selectConversation}
        onDeleteConversation={deleteConversation}
        onClose={() => setMenuOpen(false)}
      />

      <section className="main-app">
        <Topbar
          onOpenMenu={() => setMenuOpen(true)}
          onNewChat={startNewChat}
          theme={theme}
          onToggleTheme={toggleTheme}
          sidebarCollapsed={sidebarCollapsed}
          onToggleSidebar={() => setSidebarCollapsed((prev) => !prev)}
          sourcePanelOpen={sourcePanelOpen}
          onToggleSourcePanel={() => setSourcePanelOpen((prev) => !prev)}
          sourceCount={activeSources.length}
          backendStatus={backendStatus}
        />

        <div className="workspace">
          <section className="conversation">
            <div className="conversation-scroll">
              {activeConversationId === null ? (
                <div className="hero-welcome">
                  <div className="eyebrow">
                    <span className="eyebrow-emblem">UTC</span>
                    <span className="eyebrow-divider">•</span>
                    <b>TRỢ LÝ QUY CHẾ & HỌC VỤ K67</b>
                  </div>

                  <h1 className="hero-title">
                    Chào bạn, mình là <span className="hero-title-highlight">{APP_NAME}</span>
                  </h1>

                  <p className="intro">
                    Hệ thống AI hỗ trợ sinh viên <strong>Trường Đại học Giao thông Vận tải</strong> tra cứu chính xác
                    quy chế đào tạo tín chỉ, học bổng, điểm rèn luyện và điều kiện tốt nghiệp từ Sổ tay sinh viên K67.
                  </p>

                  <div className="suggestions-grid">
                    {SUGGESTIONS.map((item, idx) => (
                      <button
                        type="button"
                        key={idx}
                        className="suggestion-card"
                        onClick={() => handleQuickPrompt(item.query)}
                      >
                        <div className="suggestion-card-top">
                          <span className="suggestion-category">{item.category}</span>
                          <span className="suggestion-badge">{item.badge}</span>
                        </div>
                        <h3 className="suggestion-title">{item.title}</h3>
                        <p className="suggestion-desc">{item.desc}</p>
                        <div className="suggestion-action">
                          <span>Nhấn để hỏi</span>
                          <span className="suggestion-arrow">→</span>
                        </div>
                      </button>
                    ))}
                  </div>
                </div>
              ) : (
                (() => {
                  if (!activeConversation) return null;
                  return (
                    <div className="messages-stream">
                      {activeConversation.messages.map((message, index) => {
                        const prevMessage = index > 0 ? activeConversation.messages[index - 1] : undefined;
                        const msgTimestamp = message.timestamp ?? activeConversation.createdAt;
                        const prevTimestamp = prevMessage?.timestamp ?? activeConversation.createdAt;
                        const showDateDivider = index === 0 || !isSameCalendarDay(msgTimestamp, prevTimestamp);

                        return (
                          <Fragment key={`${activeConversation.id}-${index}`}>
                            {showDateDivider && (
                              <div className="date-divider">
                                <span>HỌC VỤ & QUY CHẾ UTC • {formatDateLabel(msgTimestamp)}</span>
                              </div>
                            )}
                            {message.role === "user" ? (
                              <div className="user-message">
                                <div className="user-message-content">
                                  <p>{message.content}</p>
                                  {message.status === "failed" && (
                                    <div className="message-error" role="alert">
                                      <span>Không thể gửi câu hỏi. Vui lòng kiểm tra lại backend.</span>
                                      <button
                                        type="button"
                                        disabled={loading}
                                        data-message-index={index}
                                        onClick={retryQuestion}
                                      >
                                        Thử lại
                                      </button>
                                    </div>
                                  )}
                                </div>
                              </div>
                            ) : (
                              <AssistantAnswer answer={message.content} sources={message.sources} />
                            )}
                          </Fragment>
                        );
                      })}
                      {loading && (
                        <div className="loading-message-card">
                          <div className="loading-pulse-dot" />
                          <div className="loading-copy">
                            <strong>{APP_NAME} đang đối chiếu dữ liệu...</strong>
                            <span>Đang trích xuất quy chế và điều khoản từ Sổ tay K67 UTC</span>
                          </div>
                        </div>
                      )}
                      <div ref={messagesEndRef} />
                    </div>
                  );
                })()
              )}
            </div>

            <form className="composer-area" onSubmit={submitQuestion}>
              <div className="composer">
                <input
                  aria-label={`Câu hỏi cho ${APP_NAME}`}
                  placeholder="Nhập câu hỏi về học vụ, đăng ký tín chỉ, xét tốt nghiệp, học bổng..."
                  value={question}
                  onChange={(event) => setQuestion(event.target.value)}
                />
                <button
                  type="submit"
                  aria-label="Gửi câu hỏi"
                  disabled={!question.trim() || loading}
                  className={`composer-send-btn ${question.trim() ? "has-text" : ""}`}
                >
                  <Icon src={ICONS.send} />
                </button>
              </div>
              <p className="composer-disclaimer">
                {APP_NAME} sử dụng nội dung Sổ tay Sinh viên K67 để đối chiếu. Luôn kiểm tra thông báo chính thức tại Cổng thông tin đào tạo UTC.
              </p>
            </form>
          </section>

          <SourcePanel
            sources={activeSources}
            open={sourcePanelOpen}
            onClose={() => setSourcePanelOpen(false)}
          />
        </div>
      </section>
    </main>
  );
}
