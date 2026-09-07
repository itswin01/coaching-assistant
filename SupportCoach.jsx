import React, { useState, useMemo, useRef, useEffect } from "react";

/* ------------------------------------------------------------------ */
/*  Mock analysis engine                                               */
/*  Stands in for the notebook's model calls. Keyword-driven so that  */
/*  typing different queries produces different, sensible output.      */
/*  NOT live inference — this is a UI demo of the interface.           */
/* ------------------------------------------------------------------ */

const ORDERS = {
  "WORD-88221": {
    order_id: "WORD-88221",
    product: 'Pro Laptop 16"',
    status: "refund_pending",
    days_since_refund_request: 20,
    refund_amount: "$1,499.00",
    customer_tier: "Pro",
  },
  "WORD-10233": {
    order_id: "WORD-10233",
    product: "Wireless Mouse",
    status: "shipped",
    days_since_order: 3,
    tracking: "TRK-556677",
    customer_tier: "Standard",
  },
};

// Fake FAQ corpus for the match cards. Keyed by issue type so a
// roughly on-topic query surfaces a plausible card, and anything
// off-topic degrades to "no strong match — escalate".
const FAQ_LIBRARY = {
  refund: [
    { id: "FAQ-2021", title: "Refund processing time", body: "Approved refunds complete in 5–7 business days.", match: 94 },
    { id: "FAQ-2044", title: "Refund delayed past 10 days", body: "Escalate to the billing team with the order ID.", match: 88 },
  ],
  shipping: [
    { id: "FAQ-1103", title: "Standard vs express shipping", body: "Standard 3–5 business days; express 1–2.", match: 91 },
    { id: "FAQ-1150", title: "Where is my tracking number", body: "Tracking is emailed when the order ships.", match: 79 },
  ],
  damaged: [
    { id: "FAQ-3300", title: "Item arrived damaged", body: "Offer replacement or full refund; photo only for carrier claims.", match: 96 },
  ],
  technical: [
    { id: "FAQ-4200", title: "Hardware fault under warranty", body: "Route to technical support for diagnostic; 12-month coverage.", match: 85 },
  ],
  account: [
    { id: "FAQ-5000", title: "Account change verification", body: "Verify identity first; never request full passwords.", match: 82 },
  ],
};

const POLICY_TEXT = {
  refund: "Approved refunds are processed within 5–7 business days. Refunds pending beyond 10 business days should be escalated to billing.",
  shipping: "Standard shipping is 3–5 business days; express is 1–2. Tracking is emailed once the order ships.",
  damaged: "For damaged items, offer a replacement or full refund. Request a photo only if needed for a carrier claim.",
  technical: "Warranty covers manufacturing defects for 12 months. Route hardware faults to technical support.",
  account: "Verify identity before any account change. Never ask for full passwords.",
};

const ISSUE_KEYWORDS = {
  refund: ["refund", "money back", "reimburse", "charge back", "my money"],
  shipping: ["shipping", "ship", "delivery", "deliver", "tracking", "arrive", "package", "where is my order"],
  damaged: ["damaged", "broken", "defective", "cracked", "shattered", "not working out of the box"],
  technical: ["not working", "won't turn on", "error", "bug", "crash", "fault", "warranty"],
  account: ["account", "password", "login", "log in", "sign in", "email address", "reset"],
};

function detectIssue(text) {
  const t = text.toLowerCase();
  let best = null;
  let bestHits = 0;
  for (const [issue, kws] of Object.entries(ISSUE_KEYWORDS)) {
    const hits = kws.filter((k) => t.includes(k)).length;
    if (hits > bestHits) {
      bestHits = hits;
      best = issue;
    }
  }
  return best; // may be null
}

function detectOrderId(text) {
  const m = text.match(/WORD-\d{3,6}/i);
  return m ? m[0].toUpperCase() : null;
}

const NEG_WORDS = ["ridiculous", "unacceptable", "furious", "angry", "worst", "terrible", "immediately", "still", "again", "!!", "!"];
const POS_WORDS = ["thanks", "thank you", "appreciate", "great", "please", "kindly"];

function analyze(text) {
  const issue_type = detectIssue(text);
  const order_id = detectOrderId(text);
  const t = text.toLowerCase();

  const negHits = NEG_WORDS.filter((w) => t.includes(w)).length + (text.match(/!/g) || []).length * 0.5;
  const posHits = POS_WORDS.filter((w) => t.includes(w)).length;

  let sentiment = "neutral";
  if (negHits >= 2 && negHits > posHits) sentiment = "negative";
  else if (posHits > negHits && negHits < 1) sentiment = "positive";

  // Urgency from intensity + time pressure words
  const urgentSignals = ["immediately", "now", "asap", "urgent", "right now", "20 days", "still waiting", "how much longer"];
  const uHits = urgentSignals.filter((w) => t.includes(w)).length + (text.match(/!/g) || []).length * 0.4;
  let urgency = "low";
  if (uHits >= 2) urgency = "high";
  else if (uHits >= 1) urgency = "medium";

  // Escalation risk = function of sentiment + urgency
  let escalation_risk = "low";
  const score = (sentiment === "negative" ? 2 : sentiment === "neutral" ? 1 : 0) + (urgency === "high" ? 2 : urgency === "medium" ? 1 : 0);
  if (score >= 3) escalation_risk = "high";
  else if (score >= 2) escalation_risk = "medium";

  // Key issue: short human phrase
  const key_issue =
    issue_type === "refund" ? "Refund delay / money-back request"
    : issue_type === "shipping" ? "Shipping / delivery status"
    : issue_type === "damaged" ? "Damaged item on arrival"
    : issue_type === "technical" ? "Product not working / warranty"
    : issue_type === "account" ? "Account access / change"
    : "General inquiry — needs classification";

  const order = order_id ? ORDERS[order_id] || null : null;
  const faqs = issue_type ? FAQ_LIBRARY[issue_type] : null;
  const policy = issue_type ? POLICY_TEXT[issue_type] : null;

  return { sentiment, urgency, escalation_risk, key_issue, order_id, issue_type, order, faqs, policy };
}

// Build a grounded suggested reply from the analysis + facts.
function draftReply(text, a) {
  const empathyOpen =
    a.sentiment === "negative"
      ? "I'm sorry for the frustration here, and you're right to expect better."
      : "Thanks for reaching out — happy to help with this.";

  let factLine = "";
  if (a.issue_type === "refund" && a.order) {
    factLine = ` I can see the refund on ${a.order.order_id} (${a.order.refund_amount}) has been pending for ${a.order.days_since_refund_request} days. Approved refunds normally complete in 5–7 business days, so this is past the expected window and I'm escalating it to our billing team now.`;
  } else if (a.issue_type === "refund") {
    factLine = " Approved refunds normally complete in 5–7 business days. I'll pull up your specific order and check exactly where it stands.";
  } else if (a.issue_type === "shipping" && a.order) {
    factLine = ` Your order ${a.order.order_id} shipped and is on tracking ${a.order.tracking}. Standard delivery runs 3–5 business days.`;
  } else if (a.issue_type === "shipping") {
    factLine = " Standard delivery runs 3–5 business days and tracking is emailed once an order ships. I'll confirm your order's status.";
  } else if (a.issue_type === "damaged") {
    factLine = " For a damaged item I can arrange a replacement or a full refund right away — whichever you prefer.";
  } else if (a.issue_type === "technical") {
    factLine = " This sounds like it may be a warranty-covered fault. I can start a quick diagnostic and, if it's a defect, arrange a repair or replacement under the 12-month warranty.";
  } else if (a.issue_type === "account") {
    factLine = " I can help with that once I've verified your identity — I'll never ask for your full password.";
  } else {
    factLine = " Let me get a couple of details so I can point you to the right fix.";
  }

  const close =
    a.escalation_risk === "high"
      ? " I'll stay on this personally until it's resolved."
      : " Let me know if anything else would help.";

  return `${empathyOpen}${factLine}${close}`;
}

/* ------------------------------------------------------------------ */
/*  Small presentational pieces                                        */
/* ------------------------------------------------------------------ */

const SIGNAL = {
  high: "#e05252",
  negative: "#e05252",
  medium: "#e0a83f",
  neutral: "#8b9aa8",
  low: "#3fb37f",
  positive: "#3fb37f",
};

function RiskCell({ label, value }) {
  const color = SIGNAL[value?.toLowerCase()] || "#8b9aa8";
  return (
    <div style={{ flex: 1, padding: "12px 14px", borderRight: "1px solid #22303c" }}>
      <div style={{ fontSize: 11, color: "#6f8090", marginBottom: 6, letterSpacing: 0.3 }}>{label}</div>
      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
        <span style={{ width: 8, height: 8, borderRadius: 8, background: color, boxShadow: `0 0 8px ${color}88` }} />
        <span style={{ fontSize: 15, color: "#e8edf2", fontWeight: 600, textTransform: "capitalize" }}>{value || "—"}</span>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  Main component                                                     */
/* ------------------------------------------------------------------ */

export default function SupportCoach() {
  const [view, setView] = useState("split"); // customer | agent | split
  const [draftInput, setDraftInput] = useState("");
  const [query, setQuery] = useState(
    "I have been waiting for my refund on order WORD-88221 for 20 days! This is ridiculous. I want my money back immediately!"
  );
  const [editedReply, setEditedReply] = useState(null);
  const [sent, setSent] = useState(false);

  const analysis = useMemo(() => (query ? analyze(query) : null), [query]);
  const suggested = useMemo(() => (analysis ? draftReply(query, analysis) : ""), [analysis, query]);

  // When a fresh query lands, reset the editable reply to the new suggestion.
  useEffect(() => {
    setEditedReply(suggested);
    setSent(false);
  }, [suggested]);

  function submitQuery() {
    if (!draftInput.trim()) return;
    setQuery(draftInput.trim());
    setDraftInput("");
    if (view === "customer") setView("split");
  }

  const showCustomer = view === "customer" || view === "split";
  const showAgent = view === "agent" || view === "split";

  return (
    <div style={styles.app}>
      <style>{globalCss}</style>

      <header style={styles.header}>
        <div>
          <div style={styles.brand}>Support Coach</div>
          <div style={styles.sub}>Live response guidance · demo</div>
        </div>
        <div style={styles.toggle}>
          {["customer", "split", "agent"].map((v) => (
            <button
              key={v}
              onClick={() => setView(v)}
              style={{
                ...styles.toggleBtn,
                ...(view === v ? styles.toggleActive : {}),
              }}
            >
              {v === "customer" ? "Customer" : v === "agent" ? "Agent" : "Both"}
            </button>
          ))}
        </div>
      </header>

      <div style={{ ...styles.stage, gridTemplateColumns: view === "split" ? "1fr 1.4fr" : "1fr" }}>
        {/* -------------------- CUSTOMER SIDE -------------------- */}
        {showCustomer && (
          <section style={styles.panel}>
            <div style={styles.panelLabel}>Customer</div>

            <div style={styles.chatArea}>
              <div style={styles.bubbleCustomer}>{query}</div>
              {sent && editedReply && (
                <div style={styles.bubbleAgent}>{editedReply}</div>
              )}
            </div>

            <div style={styles.composer}>
              <textarea
                value={draftInput}
                onChange={(e) => setDraftInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) submitQuery();
                }}
                placeholder="Type a customer message…  (⌘/Ctrl+Enter to send)"
                style={styles.composerInput}
                rows={3}
              />
              <button onClick={submitQuery} style={styles.sendBtn}>
                Send message
              </button>
            </div>
          </section>
        )}

        {/* -------------------- AGENT SIDE -------------------- */}
        {showAgent && analysis && (
          <section style={styles.panel}>
            <div style={styles.panelLabel}>Agent workspace</div>

            {/* Risk matrix */}
            <div style={styles.riskMatrix}>
              <RiskCell label="Sentiment" value={analysis.sentiment} />
              <RiskCell label="Urgency" value={analysis.urgency} />
              <RiskCell label="Escalation risk" value={analysis.escalation_risk} />
              <div style={{ flex: 1.4, padding: "12px 14px" }}>
                <div style={{ fontSize: 11, color: "#6f8090", marginBottom: 6 }}>Key issue</div>
                <div style={{ fontSize: 14, color: "#e8edf2", fontWeight: 500 }}>{analysis.key_issue}</div>
              </div>
            </div>

            {/* Two columns: grounding + reply */}
            <div style={styles.agentBody}>
              {/* Grounding column */}
              <div style={styles.groundCol}>
                <div style={styles.sectionTitle}>Matched knowledge</div>

                {analysis.faqs ? (
                  analysis.faqs.map((f) => (
                    <div key={f.id} style={styles.faqCard}>
                      <div style={styles.faqTop}>
                        <span style={styles.faqId}>{f.id}</span>
                        <span style={styles.faqMatch}>{f.match}% match</span>
                      </div>
                      <div style={styles.faqTitle}>{f.title}</div>
                      <div style={styles.faqBody}>{f.body}</div>
                    </div>
                  ))
                ) : (
                  <div style={styles.faqEmpty}>
                    No strong FAQ match. Classify the issue or escalate to a human agent.
                  </div>
                )}

                {analysis.order && (
                  <div style={styles.orderCard}>
                    <div style={styles.sectionTitle}>Order on file</div>
                    <div style={styles.orderRow}><span>Order</span><b>{analysis.order.order_id}</b></div>
                    <div style={styles.orderRow}><span>Product</span><b>{analysis.order.product}</b></div>
                    <div style={styles.orderRow}><span>Status</span><b>{analysis.order.status.replace(/_/g, " ")}</b></div>
                    {analysis.order.refund_amount && (
                      <div style={styles.orderRow}><span>Amount</span><b>{analysis.order.refund_amount}</b></div>
                    )}
                    {"days_since_refund_request" in analysis.order && (
                      <div style={styles.orderRow}><span>Pending</span><b>{analysis.order.days_since_refund_request} days</b></div>
                    )}
                    {analysis.order.tracking && (
                      <div style={styles.orderRow}><span>Tracking</span><b>{analysis.order.tracking}</b></div>
                    )}
                    <div style={styles.orderRow}><span>Tier</span><b>{analysis.order.customer_tier}</b></div>
                  </div>
                )}
              </div>

              {/* Reply column — the centerpiece */}
              <div style={styles.replyCol}>
                <div style={styles.sectionTitle}>Suggested reply · editable</div>
                <textarea
                  value={editedReply ?? ""}
                  onChange={(e) => setEditedReply(e.target.value)}
                  style={styles.replyBox}
                />
                <div style={styles.replyActions}>
                  <button
                    onClick={() => setEditedReply(suggested)}
                    style={styles.ghostBtn}
                    disabled={editedReply === suggested}
                  >
                    Reset to suggestion
                  </button>
                  <button onClick={() => setSent(true)} style={styles.primaryBtn}>
                    {sent ? "Sent ✓" : "Send to customer"}
                  </button>
                </div>
                {analysis.policy && (
                  <div style={styles.policyNote}>
                    <span style={styles.policyTag}>Grounded in policy</span>
                    {analysis.policy}
                  </div>
                )}
              </div>
            </div>
          </section>
        )}
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  Styles                                                             */
/* ------------------------------------------------------------------ */

const globalCss = `
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@500;600&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');
  * { box-sizing: border-box; }
  ::selection { background: #4a9eff44; }
  textarea { font-family: inherit; }
  textarea:focus { outline: 2px solid #4a9eff; outline-offset: 1px; }
  button:focus-visible { outline: 2px solid #4a9eff; outline-offset: 2px; }
  ::-webkit-scrollbar { width: 8px; height: 8px; }
  ::-webkit-scrollbar-thumb { background: #2a3a48; border-radius: 8px; }
`;

const styles = {
  app: {
    fontFamily: "'IBM Plex Sans', system-ui, sans-serif",
    background: "#0f1720",
    minHeight: "100vh",
    color: "#e8edf2",
    padding: "20px 22px 40px",
  },
  header: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: 18,
  },
  brand: { fontSize: 20, fontWeight: 700, letterSpacing: -0.3 },
  sub: { fontSize: 12, color: "#6f8090", marginTop: 2 },
  toggle: {
    display: "inline-flex",
    background: "#16202b",
    borderRadius: 10,
    padding: 3,
    border: "1px solid #22303c",
  },
  toggleBtn: {
    border: "none",
    background: "transparent",
    color: "#8b9aa8",
    padding: "7px 16px",
    fontSize: 13,
    fontWeight: 600,
    borderRadius: 7,
    cursor: "pointer",
    transition: "background .15s, color .15s",
  },
  toggleActive: { background: "#25384a", color: "#fff" },
  stage: { display: "grid", gap: 18, alignItems: "start" },
  panel: {
    background: "#16202b",
    border: "1px solid #22303c",
    borderRadius: 14,
    overflow: "hidden",
    minHeight: 440,
    display: "flex",
    flexDirection: "column",
  },
  panelLabel: {
    fontSize: 12,
    fontWeight: 600,
    color: "#8b9aa8",
    padding: "12px 16px",
    borderBottom: "1px solid #22303c",
    background: "#121a23",
  },
  chatArea: { flex: 1, padding: 16, display: "flex", flexDirection: "column", gap: 10, overflowY: "auto" },
  bubbleCustomer: {
    alignSelf: "flex-start",
    maxWidth: "85%",
    background: "#25384a",
    padding: "11px 14px",
    borderRadius: "4px 14px 14px 14px",
    fontSize: 14,
    lineHeight: 1.5,
  },
  bubbleAgent: {
    alignSelf: "flex-end",
    maxWidth: "85%",
    background: "#1d4e3a",
    padding: "11px 14px",
    borderRadius: "14px 4px 14px 14px",
    fontSize: 14,
    lineHeight: 1.5,
  },
  composer: { borderTop: "1px solid #22303c", padding: 12, background: "#121a23" },
  composerInput: {
    width: "100%",
    background: "#0f1720",
    border: "1px solid #2a3a48",
    borderRadius: 8,
    color: "#e8edf2",
    padding: "10px 12px",
    fontSize: 14,
    resize: "vertical",
  },
  sendBtn: {
    marginTop: 8,
    width: "100%",
    background: "#4a9eff",
    color: "#08111a",
    border: "none",
    borderRadius: 8,
    padding: "10px",
    fontSize: 14,
    fontWeight: 700,
    cursor: "pointer",
  },
  riskMatrix: {
    display: "flex",
    borderBottom: "1px solid #22303c",
    background: "#121a23",
  },
  agentBody: { display: "grid", gridTemplateColumns: "1fr 1.25fr", flex: 1, minHeight: 0 },
  groundCol: { padding: 16, borderRight: "1px solid #22303c", overflowY: "auto" },
  replyCol: { padding: 16, display: "flex", flexDirection: "column" },
  sectionTitle: { fontSize: 12, fontWeight: 600, color: "#6f8090", marginBottom: 10 },
  faqCard: {
    background: "#0f1720",
    border: "1px solid #22303c",
    borderRadius: 10,
    padding: 12,
    marginBottom: 10,
  },
  faqTop: { display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 6 },
  faqId: { fontFamily: "'IBM Plex Mono', monospace", fontSize: 12, color: "#4a9eff", fontWeight: 600 },
  faqMatch: { fontFamily: "'IBM Plex Mono', monospace", fontSize: 11, color: "#3fb37f", fontWeight: 600 },
  faqTitle: { fontSize: 14, fontWeight: 600, marginBottom: 3 },
  faqBody: { fontSize: 13, color: "#9fb0be", lineHeight: 1.45 },
  faqEmpty: {
    background: "#231a1a",
    border: "1px solid #4a2c2c",
    borderRadius: 10,
    padding: 14,
    fontSize: 13,
    color: "#e0a0a0",
    lineHeight: 1.5,
  },
  orderCard: {
    marginTop: 6,
    background: "#0f1720",
    border: "1px solid #22303c",
    borderRadius: 10,
    padding: 12,
  },
  orderRow: {
    display: "flex",
    justifyContent: "space-between",
    fontSize: 13,
    padding: "4px 0",
    color: "#9fb0be",
  },
  replyBox: {
    flex: 1,
    minHeight: 150,
    background: "#0f1720",
    border: "1px solid #3a5570",
    borderRadius: 10,
    color: "#eef3f7",
    padding: "14px 16px",
    fontSize: 15,
    lineHeight: 1.6,
    resize: "vertical",
  },
  replyActions: { display: "flex", gap: 10, marginTop: 12 },
  ghostBtn: {
    background: "transparent",
    border: "1px solid #2a3a48",
    color: "#8b9aa8",
    borderRadius: 8,
    padding: "9px 14px",
    fontSize: 13,
    fontWeight: 600,
    cursor: "pointer",
  },
  primaryBtn: {
    flex: 1,
    background: "#3fb37f",
    color: "#06140d",
    border: "none",
    borderRadius: 8,
    padding: "9px 14px",
    fontSize: 14,
    fontWeight: 700,
    cursor: "pointer",
  },
  policyNote: {
    marginTop: 14,
    fontSize: 12,
    color: "#7f909e",
    lineHeight: 1.5,
    borderTop: "1px solid #22303c",
    paddingTop: 12,
  },
  policyTag: {
    display: "inline-block",
    fontSize: 11,
    fontWeight: 600,
    color: "#3fb37f",
    background: "#12261c",
    padding: "2px 8px",
    borderRadius: 6,
    marginRight: 8,
  },
};
